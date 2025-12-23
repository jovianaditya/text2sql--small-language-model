import json
import re
import requests
from typing import Any, Dict, Optional
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain_ollama import ChatOllama
from .config import CHAT_MODEL, OLLAMA_BASE_URL

def build_llm(model: str | None = None, base_url: str | None = None, temp: float = 0.0) -> ChatOllama:
    return ChatOllama(
        model=(model or CHAT_MODEL),
        base_url=(base_url or OLLAMA_BASE_URL),
        temperature=temp,
    )

def try_llm_json_three_times(llm: ChatOllama, base_prompt: str) -> Optional[Dict[str, Any]]:
    prompts = [
        base_prompt,
        base_prompt + "\n\nReturn ONLY the JSON object. Do not add explanation.",
        base_prompt
        + "\n\nReturn ONLY JSON with keys \"sql\" and \"notes\". No extra text. "
          "Example: {\"sql\":\"SELECT 1;\",\"notes\":\"No joins used.\"}",
    ]
    for p in prompts:
        resp = llm.invoke(p)
        content = getattr(resp, "content", None) or str(resp)
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and ("sql" in obj or "notes" in obj):
                return obj
        except Exception:
            m = re.search(r"\{.*\}", content, flags=re.S)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    if isinstance(obj, dict) and ("sql" in obj or "notes" in obj):
                        return obj
                except Exception:
                    pass
    return None

def regenerate_with_extra_constraint(
    llm: ChatOllama,
    make_prompt_fn,
    ctx: str,
    question: str,
    extra_msg: str,
) -> tuple[str, str]:
    strict_prompt = make_prompt_fn(ctx + "\n\n" + extra_msg, question)
    obj = try_llm_json_three_times(llm, strict_prompt)
    if obj is None:
        return "", ""
    draft2 = (obj.get("sql") or "").strip()
    notes2 = (obj.get("notes") or "").strip()
    return draft2, notes2

# --- FIXED: OllamaEmbeddingFunction (indent + clean config methods) ---
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, base_url: str, model: str = "bge-m3:latest", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = int(timeout)

    def __call__(self, texts: Documents) -> Embeddings:
        out: Embeddings = []
        for t in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": t},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            out.append(resp.json()["embedding"])
        return out

    @classmethod
    def name(cls) -> str:
        return "ollama"

    @classmethod
    def build_from_config(cls, config: dict) -> "OllamaEmbeddingFunction":
        return cls(
            base_url=str(config.get("base_url", "http://localhost:11434")),
            model=str(config.get("model", "bge-m3:latest")),
            timeout=int(config.get("timeout", 60)),
        )

    # new: allow Chroma to serialize EF config
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name(),
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
        }