# RAG on AWS Bedrock with Aurora PostgreSQL

This document summarises the architecture proposed for the Bedrock-based Retrieval Augmented Generation system. The flow mirrors the ideas present in `rag.py`, but replaces OpenSearch with Aurora PostgreSQL (pgvector) and adds deployment guidance for an EC2-hosted API.

## High-level components

1. **Raw data storage** – Documents live in Amazon S3. The ingestion job reads from a configurable bucket/prefix and supports common text/PDF formats.
2. **Chunking & embedding** – LangChain drives text splitting and batching. Embeddings come from a Bedrock embedding foundation model (default: `amazon.titan-embed-text-v2`).
3. **Vector store** – Aurora PostgreSQL with the `pgvector` extension stores embeddings. LangChain's `PGVector` wrapper manages schema and metadata. Each chunk keeps S3 object provenance to simplify debugging.
4. **Retrieval & fusion** – A LangChain-based retriever executes similarity search. Optional query rewriting (standalone question rewriter + multi-query fusion) and reranking (Bedrock Cohere Rerank) preserve the original `rag.py` capabilities.
5. **Orchestration** – A FastAPI service exposes a `/query` endpoint. At runtime it loads the retriever + Bedrock LLM (default: `anthropic.claude-3-haiku-20240307-v1:0`) to produce answers in Vietnamese.

## Execution modes

- **Local development** – Run the ingestion CLI (`python -m src.ingest.pipeline`) to populate Aurora, then start the API (`uvicorn src.api.app:app --reload`). Requires connectivity to Aurora (can be provided via IAM auth or username/password) and AWS credentials with Bedrock + S3 permissions.
- **AWS deployment** – Build the included Docker image and run it on an EC2 instance in the same VPC as the Aurora cluster. Environment variables configure database, S3 and Bedrock models. A systemd unit file example is included in the README for production hardening.

## Data model & metadata

- `collection_name` identifies logical datasets. Each row includes `source_key`, `source_bucket`, `chunk_id`, `chunk_order`, and optional `tags` for hybrid filtering.
- The ingestion pipeline supports idempotent updates by deleting existing rows for the same `collection_name` before inserting fresh embeddings (configurable).

## Retry and observability

- Boto3 retries S3 downloads automatically. Bedrock invocations use exponential backoff (via botocore defaults).
- The FastAPI app logs retrieved chunk metadata alongside answer latency. Structured logs can be shipped to CloudWatch when running on EC2.

