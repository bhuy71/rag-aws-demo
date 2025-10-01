"""Centralised application configuration using environment variables."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    aws_region: str = Field(validation_alias="AWS_REGION")
    bedrock_region: Optional[str] = Field(default=None, validation_alias="BEDROCK_REGION")
    bedrock_chat_model_id: str = Field(
        default="anthropic.claude-3-haiku-20240307-v1:0",
        validation_alias="BEDROCK_CHAT_MODEL_ID",
    )
    bedrock_embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2",
        validation_alias="BEDROCK_EMBEDDING_MODEL_ID",
    )
    bedrock_rerank_model_id: Optional[str] = Field(
        default=None,
        validation_alias="BEDROCK_RERANK_MODEL_ID",
    )

    s3_bucket: str = Field(validation_alias="RAG_S3_BUCKET")
    s3_prefix: str = Field(default="", validation_alias="RAG_S3_PREFIX")

    pg_host: str = Field(validation_alias="PG_HOST")
    pg_port: int = Field(default=5432, validation_alias="PG_PORT")
    pg_user: str = Field(validation_alias="PG_USER")
    pg_password: str = Field(validation_alias="PG_PASSWORD")
    pg_database: str = Field(validation_alias="PG_DATABASE")
    pg_use_ssl: bool = Field(default=True, validation_alias="PG_USE_SSL")
    pg_require_iam: bool = Field(default=False, validation_alias="PG_REQUIRE_IAM")

    vector_collection: str = Field(default="rag_docs", validation_alias="PG_VECTOR_COLLECTION")
    vector_search_k: int = Field(default=8, validation_alias="VECTOR_SEARCH_K")
    vector_search_k_rerank: int = Field(default=4, validation_alias="VECTOR_SEARCH_K_RERANK")

    chunk_size: int = Field(default=1200, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, validation_alias="CHUNK_OVERLAP")

    enable_query_fusion: bool = Field(default=True, validation_alias="ENABLE_QUERY_FUSION")
    fusion_variant_count: int = Field(default=3, validation_alias="FUSION_VARIANT_COUNT")
    enable_hyde: bool = Field(default=False, validation_alias="ENABLE_HYDE")

    answer_language: str = Field(default="vi", validation_alias="ANSWER_LANGUAGE")

    @property
    def resolved_bedrock_region(self) -> str:
        return self.bedrock_region or self.aws_region

    @property
    def pg_connection_uri(self) -> str:
        scheme = "postgresql+psycopg2"
        sslmode = "require" if self.pg_use_ssl else "prefer"
        return (
            f"{scheme}://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}?sslmode={sslmode}"
        )


def _sync_aws_region(settings: Settings) -> None:
    """Ensure boto3 sees the same region as our settings."""

    os.environ.setdefault("AWS_DEFAULT_REGION", settings.aws_region)
    if "AWS_REGION" not in os.environ:
        os.environ["AWS_REGION"] = settings.aws_region


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    settings = Settings()
    _sync_aws_region(settings)
    return settings

