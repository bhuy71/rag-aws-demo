"""High level LangChain pipeline orchestrating the RAG flow."""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

from src.config import Settings, get_settings
from src.rag.retriever import AuroraPGVectorRetriever, build_retriever
from src.shared import bedrock
from src.shared.prompts import PromptRepo

LOGGER = logging.getLogger(__name__)


def _cleanup_result_tag(text: str) -> str:
    match = re.search(r"<result>(.*?)</result>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _format_history(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for turn in history:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_contexts(docs: List[Document]) -> str:
    blocks: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        score = doc.metadata.get("rerank_score")
        prefix = f"[{idx}] {source}"
        if score is not None:
            prefix += f" (score={score:.3f})"
        blocks.append(f"{prefix}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


class RAGPipeline:
    """Encapsulates the RAG pipeline including question rewriting and answering."""

    def __init__(
        self,
        retriever: AuroraPGVectorRetriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or build_retriever(self.settings)
        self.answer_llm = bedrock.get_chat_model(self.settings.bedrock_chat_model_id)
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PromptRepo.get_system_prompt()),
                ("human", PromptRepo.get_human_prompt()),
            ]
        )

    def _rewrite_question(
        self, question: str, history: Optional[List[Dict[str, str]]]
    ) -> str:
        if not history:
            return question
        history_text = _format_history(history)
        if not history_text:
            return question
        system_prompt = PromptRepo.get_contextualize_system_prompt()
        user_prompt = (
            f"Conversation so far:\n{history_text}\n\n"
            f"Create a standalone question for: <question>{question}</question>"
        )
        try:
            response = bedrock.invoke_text_generation(
                model_id=self.settings.bedrock_chat_model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:  # pragma: no cover - network call
            LOGGER.warning("Question rewrite failed: %s", exc)
            return question
        rewritten = _cleanup_result_tag(response)
        LOGGER.debug("Rewritten question: %s", rewritten)
        return rewritten or question

    def _build_context(self, docs: List[Document]) -> str:
        if not docs:
            return ""
        return _format_contexts(docs)

    def run(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, object]:
        LOGGER.info("Running RAG pipeline")
        rewritten_question = self._rewrite_question(question, history)
        retrieved_docs = self.retriever.get_relevant_documents(rewritten_question)
        context_text = self._build_context(retrieved_docs)
        messages = self.answer_prompt.format_messages(
            contexts=context_text,
            question=rewritten_question,
        )
        response = self.answer_llm.invoke(messages)
        answer_text = getattr(response, "content", "")
        if isinstance(answer_text, list):
            answer_text = "".join(
                segment.get("text", "")
                for segment in answer_text
                if isinstance(segment, dict)
            )
        context_payload = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in retrieved_docs
        ]
        result = {
            "question": question,
            "rewritten_question": rewritten_question,
            "answer": answer_text.strip(),
            "contexts": context_payload,
            "query_variants": self.retriever.last_query_variants,
            "hyde_document": self.retriever.last_hypothetical_document,
        }
        return result


__all__ = ["RAGPipeline"]

