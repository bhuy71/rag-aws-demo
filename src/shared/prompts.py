"""Prompt repository reused across Lambdas."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptRepo:
    """Collection of prompt templates aligned with the original rag.py flow."""

    @staticmethod
    def get_contextualize_system_prompt() -> str:
        """Prompt for rewriting follow-up questions."""
        return (
            "Referring to the conversation history, create a standalone version of the "
            "<question>. The rewritten question must keep all critical terms. "
            "Wrap the result in <result> tags. If the conversation is irrelevant, "
            "return the original question inside <result> tags without modifications."
        )

    @staticmethod
    def get_system_prompt() -> str:
        """Prompt driving the answering behaviour."""
        return (
            "You are a senior assistant that answers user questions in Vietnamese. "
            "You will receive background context made of text chunks produced during "
            "a retrieval step. Read the context carefully and provide the most "
            "accurate answer possible. If the context is insufficient, reply with "
            "'Khong co thong tin phu hop trong ngu canh.'"
        )

    @staticmethod
    def get_human_prompt() -> str:
        """Prompt for formatting contexts and question."""
        return (
            "Ben duoi la tap cac ngu canh: <contexts>{contexts}</contexts>\n\n"
            "Tra loi cau hoi sau bang tieng Viet, khong them loi mo dau."
            " Neu khong tim thay thong tin phu hop trong ngu canh hay tra loi"
            " dung nhu sau: 'Khong co thong tin phu hop trong ngu canh.'\n\n"
            "<question>{question}</question>"
        )

    @staticmethod
    def get_rag_fusion_prompt() -> str:
        """Prompt used to generate query variants for RAG Fusion."""
        return (
            "Ban se nhan mot cau hoi bang tieng Viet. Hay tao toi da 3 truy van "
            "tim kiem khac nhau nhung giu nguyen nghia cua cau hoi ban dau. "
            "Moi truy van viet tren mot dong, khong danh so thu tu."
        )

    @staticmethod
    def get_hyde_prompt() -> str:
        """Prompt for generating hypothetical answers used in HyDE."""
        return (
            "Viet mot doan van ngan gom 3-5 cau tra loi cau hoi sau day du tren "
            "toan bo kien thuc ban co. Su dung giong dieu thong tin, khong gioi han"
            " thong tin. Cau hoi: {question}"
        )
