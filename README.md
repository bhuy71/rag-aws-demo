# Bedrock RAG on Aurora PostgreSQL

This project provides a Retrieval Augmented Generation (RAG) chatbot pipeline built with LangChain, Amazon Bedrock foundation models, raw content living in Amazon S3, and embeddings stored in Aurora PostgreSQL (pgvector). It reproduces the flow of the original `rag.py` script while swapping OpenSearch for Aurora and exposing a FastAPI service that can be deployed to EC2.

## Features

- S3 loaders + recursive text splitting for chunking raw documents.
- Bedrock Titan embeddings stored in Aurora PostgreSQL via the pgvector extension.
- Query rewriting, multi-query fusion, optional HyDE augmentation, and Cohere reranking.
- FastAPI endpoint (`POST /query`) returning the LLM answer plus retrieved chunks.
- Dockerfile for EC2 deployment and scripts for initial Aurora provisioning.

## Repository layout

```
+-- src
¦   +-- api/app.py            # FastAPI app exposing the RAG endpoint
¦   +-- config.py             # Pydantic settings and helpers
¦   +-- ingest/pipeline.py    # CLI to pull content from S3 and persist embeddings
¦   +-- rag/pipeline.py       # High-level orchestration (rewrite ? retrieve ? answer)
¦   +-- rag/retriever.py      # Aurora pgvector retriever with fusion/HyDE/rerank
¦   +-- shared                # Bedrock + prompt utilities reused across modules
+-- scripts/aurora_init.sql   # Optional SQL to provision pgvector tables
+-- docs/architecture.md      # Design notes
+-- Dockerfile                # Container for EC2
+-- requirements.txt
```

## Prerequisites

- Python 3.11+
- AWS account with access to Amazon Bedrock, S3, and Aurora PostgreSQL
- An Aurora PostgreSQL cluster (>= 15.2) with the `vector` extension enabled
- S3 bucket containing the raw documents you want to expose through the chatbot
- AWS credentials configured locally (via `~/.aws/credentials`, environment vars, or SSO)

## Configuration

All runtime configuration is handled through environment variables. You can place them in an `.env` file during local development. Key variables:

| Variable | Description |
| --- | --- |
| `AWS_REGION` | Default AWS region for Bedrock/S3 | 
| `BEDROCK_CHAT_MODEL_ID` | Bedrock chat model (default `anthropic.claude-3-haiku-20240307-v1:0`) |
| `BEDROCK_EMBEDDING_MODEL_ID` | Bedrock embedding model (`amazon.titan-embed-text-v2`) |
| `BEDROCK_RERANK_MODEL_ID` | Optional reranker (e.g. `cohere.rerank-v3`) |
| `RAG_S3_BUCKET` / `RAG_S3_PREFIX` | Bucket and prefix containing raw docs |
| `PG_HOST`, `PG_PORT`, `PG_DATABASE`, `PG_USER`, `PG_PASSWORD` | Aurora connection info |
| `PG_VECTOR_COLLECTION` | Logical collection name inside pgvector (default `rag_docs`) |
| `VECTOR_SEARCH_K` / `VECTOR_SEARCH_K_RERANK` | Retriever top-k candidates |
| `ENABLE_QUERY_FUSION` | Toggle multi-query fusion (default `true`) |
| `ENABLE_HYDE` | Toggle HyDE augmentation (default `false`) |

> **Aurora authentication**: if you rely on IAM database authentication, set `PG_PASSWORD` to a generated token before starting the API.

## Prepare Aurora PostgreSQL

1. Connect to the Aurora cluster using `psql`.
2. Enable the `vector` extension and (optionally) pre-create LangChain tables:

   ```sql
   \i scripts/aurora_init.sql
   ```

3. Grant your database user `INSERT`, `UPDATE`, `DELETE`, and `SELECT` on `langchain_pg_collection` and `langchain_pg_embedding`.

## Ingest documents locally

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Export environment variables or create an .env file first
python -m src.ingest.pipeline --prefix my/documents/prefix --collection rag-demo
```

This command downloads compatible files from `s3://$RAG_S3_BUCKET/my/documents/prefix`, splits them into chunks, embeds each chunk with Bedrock, and persists them into the configured Aurora collection. Use `--no-reset` to append to an existing collection instead of recreating it.

## Run the API locally

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8080
```

Send a request:

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Cho toi biet thong tin san pham ABC", "history": []}'
```

The response JSON includes the answer, rewritten question, the chunks used to construct the answer, query variants produced during fusion, and the HyDE synthetic document (if enabled).

## Container build & EC2 deployment

1. **Build and push the image**

   ```bash
   docker build -t bedrock-rag:latest .
   aws ecr create-repository --repository-name bedrock-rag --region $AWS_REGION
   docker tag bedrock-rag:latest <account-id>.dkr.ecr.$AWS_REGION.amazonaws.com/bedrock-rag:latest
   aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin <account-id>.dkr.ecr.$AWS_REGION.amazonaws.com
   docker push <account-id>.dkr.ecr.$AWS_REGION.amazonaws.com/bedrock-rag:latest
   ```

2. **Provision infrastructure**

   - Place the EC2 instance in the same VPC/subnet as the Aurora cluster for low latency.
   - Attach an instance role with Bedrock `InvokeModel`, S3 `GetObject`, and Aurora `rds-db:connect` permissions.
   - Store required environment variables in `/etc/bedrock-rag.env` (or use AWS Systems Manager Parameter Store/Secrets Manager).

3. **Run the container on EC2**

   ```bash
   docker run -d --name bedrock-rag \
     --env-file /etc/bedrock-rag.env \
     -p 8080:8080 \
     <account-id>.dkr.ecr.$AWS_REGION.amazonaws.com/bedrock-rag:latest
   ```

   For production, create a `systemd` unit:

   ```ini
   [Unit]
   Description=Bedrock RAG API
   After=docker.service
   Requires=docker.service

   [Service]
   Restart=always
   ExecStart=/usr/bin/docker run --rm --name bedrock-rag \
     --env-file /etc/bedrock-rag.env -p 8080:8080 \
     <account-id>.dkr.ecr.$AWS_REGION.amazonaws.com/bedrock-rag:latest
   ExecStop=/usr/bin/docker stop bedrock-rag

   [Install]
   WantedBy=multi-user.target
   ```

   Enable it with `sudo systemctl enable --now bedrock-rag.service`.

4. **Expose the API**

   - Optionally place an Application Load Balancer or API Gateway in front of EC2 for TLS termination.
   - Use AWS WAF or Cognito for authentication if the API should not be public.

## Observability & maintenance

- FastAPI logs are emitted to stdout; configure CloudWatch Logs by running the container with `awslogs` driver if desired.
- Enable AWS X-Ray or adopt structured logging if deeper tracing is needed.
- Schedule the ingestion CLI (via AWS Batch or Step Functions) whenever the S3 corpus changes.

## Troubleshooting

- **Missing pgvector**: run `CREATE EXTENSION vector;` in Aurora.
- **Bedrock permission errors**: ensure the IAM role/user has `bedrock:InvokeModel` for the chosen model IDs.
- **Slow responses**: tune `VECTOR_SEARCH_K`, `VECTOR_SEARCH_K_RERANK`, or disable HyDE/reranking.
- **Large buckets**: restrict ingestion with `--prefix` or extend `SUPPORTED_SUFFIXES` in `src/ingest/pipeline.py` to match your file types.

## Next steps

- Add Guardrails or knowledge cut-off validation before answering users.
- Integrate AWS Lambda / API Gateway if you prefer serverless inference.
- Swap the retriever to `langchain.vectorstores.AuroraPGVector` when it becomes available in core LangChain.
