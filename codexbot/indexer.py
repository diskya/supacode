from __future__ import annotations

import os
import re
import ast
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from .embedder import build_embedder, BaseEmbedder
from .store import ChromaStore, Chunk


SUPPORTED_EXT = {".py": "python"}


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _chunks_python(path: str, code: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        lines = code.splitlines()
        return [
            Chunk(
                id=str(uuid.uuid4()),
                path=path,
                start_line=1,
                end_line=len(lines),
                symbol=os.path.basename(path),
                kind="file",
                lang="python",
                text=code,
            )
        ]

    source_lines = code.splitlines()

    def add_chunk(node, symbol: str, kind: str):
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", len(source_lines))
        text = "\n".join(source_lines[start - 1 : end])
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                path=path,
                start_line=start,
                end_line=end,
                symbol=symbol,
                kind=kind,
                lang="python",
                text=text,
            )
        )

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            add_chunk(node, node.name, "function")
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            add_chunk(node, node.name, "function")
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef):
            add_chunk(node, node.name, "class")
            self.generic_visit(node)

    Visitor().visit(tree)

    if not chunks:
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                path=path,
                start_line=1,
                end_line=len(source_lines),
                symbol=os.path.basename(path),
                kind="file",
                lang="python",
                text=code,
            )
        )
    return chunks


def walk_repo(repo: str) -> Iterable[str]:
    ignore_dirs = {".git", "node_modules", ".venv", "venv", "dist", "build", "__pycache__"}
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for fn in files:
            ext = os.path.splitext(fn)[1]
            if ext in SUPPORTED_EXT:
                yield os.path.join(root, fn)


def parse_chunks(path: str) -> List[Chunk]:
    code = _read_file(path)
    ext = os.path.splitext(path)[1]
    if ext == ".py":
        return _chunks_python(path, code)
    return []


def index_repo(
    repo: str,
    index_dir: str,
    embedder_name: Optional[str] = None,
) -> None:
    store = ChromaStore(index_dir)
    store.init_empty()

    embedder: BaseEmbedder = build_embedder(embedder_name)
    
    # Store embedder metadata
    store.collection.modify(metadata={"embedder": embedder_name or "default_embedder"})

    all_chunks: List[Chunk] = []
    for path in tqdm(list(walk_repo(repo)), desc="Scanning"):
        all_chunks.extend(parse_chunks(path))

    if not all_chunks:
        print("No chunks found to index.")
        return

    texts = [c.text for c in all_chunks]
    vectors = embedder.encode(texts).tolist()

    store.add_chunks_and_vectors(all_chunks, vectors)

    print(f"Indexed {len(all_chunks)} chunks. Vector dim={len(vectors[0])} saved to {index_dir}")
