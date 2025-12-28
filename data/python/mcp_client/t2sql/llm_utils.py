from langchain_ollama import ChatOllama
from .config import CHAT_MODEL, OLLAMA_URL

def build_llm(model: str | None = None, base_url: str | None = None, temp: float = 0.0) -> ChatOllama:
    return ChatOllama(
        model=(model or CHAT_MODEL),
        base_url=(base_url or OLLAMA_URL),
        temperature=temp,
    )