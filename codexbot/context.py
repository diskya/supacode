from __future__ import annotations

from typing import Optional

from .retrieval import search, assemble_context


def build_prompt(
    index_dir: str,
    user_query: str,
    embedder_name: Optional[str] = None,
    top_k: int = 10,
    max_context_chars: int = 12000,
    threshold: float = 1.0,
) -> str:
    results = search(
        index_dir, user_query, embedder_name=embedder_name, top_k=top_k, threshold=threshold
    )
    chunks = [chunk for chunk, distance in results]
    context = assemble_context(chunks, max_chars=max_context_chars)
    # The system prompt is now handled by the LLMClient.
    # We just need to format the user query and the context.
    prompt = (
        f"USER QUESTION: {user_query}\n\n"
        f"CONTEXT:\n{context}"
    )
    return prompt
