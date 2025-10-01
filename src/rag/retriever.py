"""Custom retriever that mirrors rag.py behaviour using Aurora PGVector."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from langchain.schema import BaseRetriever, Document
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, PrivateAttr
from pydantic.config import ConfigDict

from src.config import Settings, get_settings
from src.shared import bedrock
from src.shared.prompts import PromptRepo

LOGGER = logging.getLogger(__name__)


class AuroraPGVectorRetriever(BaseRetriever):
    """Retriever with optional query fusion, HyDE and reranking."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: Settings
    vector_store: PGVector
    primary_k: int
    final_k: int
    enable_query_fusion: bool
    enable_hyde: bool
    fusion_variant_count: int
    rerank_model_id: Optional[str]
    last_query_variants: List[str] = Field(default_factory=list)
    last_hypothetical_document: Optional[str] = None

    _query_llm: object = PrivateAttr()
    _fusion_prompt: ChatPromptTemplate = PrivateAttr()
    _hyde_prompt: ChatPromptTemplate = PrivateAttr()

    def __init__(self, vector_store: PGVector, settings: Settings | None = None) -> None:  # type: ignore[override]
        st = settings or get_settings()
        super().__init__(
            settings=st,
            vector_store=vector_store,
            primary_k=st.vector_search_k,
            final_k=max(st.vector_search_k_rerank, 1),
            enable_query_fusion=st.enable_query_fusion,
            enable_hyde=st.enable_hyde,
            fusion_variant_count=max(st.fusion_variant_count, 1),
            rerank_model_id=st.bedrock_rerank_model_id,
        )
        self._query_llm = bedrock.get_chat_model(st.bedrock_chat_model_id)
        self._fusion_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Ban la tro ly tao cac truy van tim kiem tuong tu."),
                ("human", PromptRepo.get_rag_fusion_prompt()),
            ]
        )
        self._hyde_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Ban viet cac doan van mo phong cau tra loi."),
                ("human", PromptRepo.get_hyde_prompt()),
            ]
        )

    def _deduplicate(self, docs: Sequence[Document]) -> List[Document]:
        seen: Dict[str, Document] = {}
        ordered: List[Document] = []
        for doc in docs:
            key = doc.metadata.get("source", "") + str(doc.metadata.get("chunk_id", ""))
            key = key or doc.page_content[:100]
            if key not in seen:
                seen[key] = doc
                ordered.append(doc)
        return ordered

    def _to_plain_text(self, response: object) -> str:
        content = getattr(response, "content", "")
        if isinstance(content, list):
            text_segments = [
                segment.get("text", "")
                for segment in content
                if isinstance(segment, dict)
            ]
            content = "\n".join(text_segments)
        return str(content)

    def _generate_variant_queries(self, query: str) -> List[str]:
        if not self.enable_query_fusion:
            self.last_query_variants = []
            return []

        messages = self._fusion_prompt.format_messages(question=query)
        response = self._query_llm.invoke(messages)
        content = self._to_plain_text(response)
        variants: List[str] = []
        for line in content.splitlines():
            stripped = line.strip("-* ").strip()
            if not stripped:
                continue
            if stripped.lower() == query.lower():
                continue
            variants.append(stripped)
            if len(variants) >= self.fusion_variant_count:
                break
        self.last_query_variants = variants
        LOGGER.debug("Generated %d query variants", len(variants))
        return variants

    def _generate_hypothetical_document(self, query: str) -> str:
        if not self.enable_hyde:
            self.last_hypothetical_document = None
            return ""
        messages = self._hyde_prompt.format_messages(question=query)
        response = self._query_llm.invoke(messages)
        hypo = self._to_plain_text(response).strip()
        self.last_hypothetical_document = hypo
        LOGGER.debug("Generated HyDE document length=%d", len(hypo))
        return hypo

    def _retrieve_candidates(self, query: str) -> List[Document]:
        candidates = self.vector_store.similarity_search(query, k=self.primary_k)
        LOGGER.debug("Base query returned %d candidates", len(candidates))
        for variant in self._generate_variant_queries(query):
            variant_docs = self.vector_store.similarity_search(variant, k=self.primary_k)
            LOGGER.debug("Variant '%s' returned %d candidates", variant, len(variant_docs))
            candidates.extend(variant_docs)
        hypothetical = self._generate_hypothetical_document(query)
        if hypothetical:
            hyde_docs = self.vector_store.similarity_search(hypothetical, k=self.primary_k)
            LOGGER.debug("HyDE generated %d candidates", len(hyde_docs))
            candidates.extend(hyde_docs)
        return self._deduplicate(candidates)

    def _apply_rerank(self, query: str, docs: Sequence[Document]) -> List[Document]:
        if not self.rerank_model_id:
            return list(docs[: self.final_k])
        try:
            reranked = bedrock.invoke_rerank(self.rerank_model_id, query, docs, top_n=self.final_k)
            if not reranked:
                return list(docs[: self.final_k])
            return reranked
        except Exception as exc:  # pragma: no cover - network call
            LOGGER.warning("Rerank failed: %s", exc)
            return list(docs[: self.final_k])

    def _get_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        LOGGER.debug("Retrieving documents for query: %s", query)
        candidates = self._retrieve_candidates(query)
        LOGGER.debug("Total unique candidates: %d", len(candidates))
        return self._apply_rerank(query, candidates)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        return self._get_relevant_documents(query)


def build_vector_store(settings: Settings | None = None) -> PGVector:
    """Factory for PGVector backed by Aurora."""

    st = settings or get_settings()
    embedding_model = bedrock.get_embedding_model(st.bedrock_embedding_model_id)
    return PGVector(
        connection_string=st.pg_connection_uri,
        collection_name=st.vector_collection,
        embedding_function=embedding_model,
    )


def build_retriever(settings: Settings | None = None) -> AuroraPGVectorRetriever:
    """Create a retriever bound to the configured collection."""

    st = settings or get_settings()
    vector_store = build_vector_store(st)
    return AuroraPGVectorRetriever(vector_store=vector_store, settings=st)
