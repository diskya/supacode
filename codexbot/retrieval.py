from __future__ import annotations

from typing import List

from .embedder import build_embedder
from .store import ChromaStore, Chunk


def search(
    index_dir: str,
    query: str,
    embedder_name: str | None,
    top_k: int = 10,
    threshold: float = 1.5
) -> List[Tuple[Chunk, float]]:
    """
    Performs a semantic search using the ChromaDB store, with a similarity threshold.
    """
    store = ChromaStore(index_dir)
    
    if embedder_name is None:
        collection_metadata = store.collection.metadata
        if collection_metadata and "embedder" in collection_metadata:
            embedder_name = collection_metadata["embedder"]

    embedder = build_embedder(embedder_name)
    query_vector = embedder.encode([query])[0].tolist()
    
    results_with_distances = store.search(query_vector, limit=top_k)
    
    # Filter results based on the threshold.
    # ChromaDB's L2 distance is lower for more similar items.
    return [(chunk, distance) for chunk, distance in results_with_distances if distance <= threshold]


def assemble_context(chunks: List[Chunk], max_chars: int = 12000) -> str:
    """
    Assembles a context string from a list of chunks.
    """
    out_lines: List[str] = []
    used = 0
    for ch in chunks:
        header = f"// file: {ch.path}:{ch.start_line}-{ch.end_line} [{ch.kind} {ch.symbol}]"
        body = ch.text
        block = header + "\n" + body
        if used + len(block) > max_chars and out_lines:
            break
        out_lines.append(block)
        used += len(block)
        out_lines.append("\n\n")
    return "".join(out_lines)
