"""Utilities for interacting with Amazon Bedrock."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence

import boto3
from langchain.schema import Document
from langchain_aws.chat_models import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings


@lru_cache(maxsize=4)
def _bedrock_client() -> Any:
    region = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION")
    if not region:
        raise EnvironmentError("BEDROCK_REGION or AWS_REGION must be defined")
    return boto3.client("bedrock-runtime", region_name=region)


@lru_cache(maxsize=4)
def get_chat_model(model_id: str) -> ChatBedrock:
    """Return a cached Bedrock chat model."""
    return ChatBedrock(model_id=model_id, client=_bedrock_client())


@lru_cache(maxsize=4)
def get_embedding_model(model_id: str) -> BedrockEmbeddings:
    """Return a cached Bedrock embedding model."""
    return BedrockEmbeddings(model_id=model_id, client=_bedrock_client())


def invoke_text_generation(model_id: str, system_prompt: str, user_prompt: str) -> str:
    """Low-level invocation helper when LangChain wrappers are not desired."""
    response = _bedrock_client().invoke_model(
        modelId=model_id,
        body=json.dumps(
            {
                "messages": [
                    {"role": "system", "content": [{"text": system_prompt}]},
                    {"role": "user", "content": [{"text": user_prompt}]},
                ]
            }
        ),
    )
    payload = json.loads(response["body"].read())
    outputs = payload.get("output", {}).get("message", {}).get("content", [])
    texts = [item.get("text", "") for item in outputs]
    return "\n".join(texts).strip()


def invoke_rerank(
    model_id: str,
    query: str,
    documents: Sequence[Document],
    top_n: int,
) -> List[Document]:
    """Call a Bedrock rerank foundation model (e.g. Cohere Rerank).

    The rerank API expects plain texts, so metadata is reconstructed on return.
    """

    if not documents:
        return []

    payload = {
        "query": query,
        "documents": [doc.page_content for doc in documents],
        "topN": min(top_n, len(documents)),
        "returnDocuments": True,
    }

    response = _bedrock_client().invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
    )
    body = json.loads(response["body"].read())
    results = body.get("results", [])

    ranked_docs: List[Document] = []
    for result in results:
        idx = result.get("index")
        if idx is None or idx >= len(documents):
            continue
        doc = documents[idx]
        metadata = dict(doc.metadata)
        metadata["rerank_score"] = result.get("relevanceScore")
        ranked_docs.append(Document(page_content=doc.page_content, metadata=metadata))
    return ranked_docs


def chunked(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """Yield lists with at most size elements."""
    chunk: List[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def embed_texts(
    embedding_model: BedrockEmbeddings, texts: List[str], batch_size: int = 16
) -> List[List[float]]:
    """Batch embedding helper to reduce Bedrock calls."""
    results: List[List[float]] = []
    for batch in chunked(texts, batch_size):
        results.extend(embedding_model.embed_documents(batch))
    return results

