"""CLI to ingest documents from S3 into Aurora PostgreSQL (pgvector)."""
from __future__ import annotations

import argparse
import logging
from typing import List, Sequence

import boto3
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import S3DirectoryLoader

from src.config import get_settings
from src.shared import bedrock

LOGGER = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf", ".html", ".json")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def load_documents(bucket: str, prefix: str, allowed_suffixes: Sequence[str]) -> List[Document]:
    """Load documents from S3 using LangChain's S3DirectoryLoader."""

    LOGGER.info("Loading documents from s3://%s/%s", bucket, prefix or "")
    loader = S3DirectoryLoader(
        bucket=bucket,
        prefix=prefix or "",
    )
    documents = [
        doc
        for doc in loader.load()
        if doc.metadata.get("source", "").lower().endswith(tuple(allowed_suffixes))
    ]
    LOGGER.info("Loaded %d documents", len(documents))
    return documents


def chunk_documents(
    documents: Sequence[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Split documents preserving S3 provenance metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(list(documents))
    LOGGER.info("Split into %d chunks", len(chunks))

    for idx, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata.setdefault("chunk_id", idx)
        metadata.setdefault("source", metadata.get("source"))
        chunk.metadata = metadata
    return chunks


def persist_chunks(
    chunks: Sequence[Document],
    collection_name: str,
    reset_collection: bool,
) -> None:
    """Persist chunks into Aurora PostgreSQL using pgvector."""

    if not chunks:
        LOGGER.warning("No chunks to persist")
        return

    settings = get_settings()
    embedding_model = bedrock.get_embedding_model(settings.bedrock_embedding_model_id)
    connection_string = settings.pg_connection_uri

    vector_store = PGVector(
        connection_string=connection_string,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    if reset_collection:
        LOGGER.info("Resetting collection %s", collection_name)
        try:
            vector_store.delete_collection()
            # Re-instantiate to ensure tables exist after deletion
            vector_store = PGVector(
                connection_string=connection_string,
                collection_name=collection_name,
                embedding_function=embedding_model,
            )
        except Exception as exc:  # pragma: no cover - requires database
            LOGGER.warning("Failed to reset collection %s: %s", collection_name, exc)

    LOGGER.info("Persisting %d chunks into collection '%s'", len(chunks), collection_name)
    vector_store.add_documents(list(chunks))
    LOGGER.info("Completed persistence")


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Ingest S3 documents into Aurora pgvector")
    parser.add_argument("--collection", default=settings.vector_collection, help="Collection name override")
    parser.add_argument("--prefix", default=settings.s3_prefix, help="Optional S3 prefix")
    parser.add_argument("--suffixes", nargs="*", default=SUPPORTED_SUFFIXES, help="Allowed file extensions")
    parser.add_argument("--no-reset", action="store_true", help="Do not drop the existing collection before ingest")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)

    boto3.setup_default_session(region_name=settings.aws_region)

    docs = load_documents(settings.s3_bucket, args.prefix, args.suffixes)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    persist_chunks(chunks, args.collection, reset_collection=not args.no_reset)


if __name__ == "__main__":
    main()
