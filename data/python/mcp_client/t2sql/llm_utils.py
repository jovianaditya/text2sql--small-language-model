from __future__ import annotations

from langchain_ollama import ChatOllama
from .config import CHAT_MODEL, OLLAMA_URL

def build_llm_json(model: str | None = None, base_url: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=(model or CHAT_MODEL),
        base_url=(base_url or OLLAMA_URL),
        temperature=0.0,
        num_ctx=2048,
        num_predict=160,
        stop=None,
    )

def build_llm_repair(model: str | None = None, base_url: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=(model or CHAT_MODEL),
        base_url=(base_url or OLLAMA_URL),
        temperature=0.0,
        num_ctx=2048,
        num_predict=256,
        stop=None,
    )