from __future__ import annotations

import argparse
import os
import sys

from .indexer import index_repo
from .retrieval import search, assemble_context
from .context import build_prompt


def cmd_index(args: argparse.Namespace) -> None:
    index_repo(repo=args.repo, index_dir=args.index_dir, embedder_name=args.embedder)


def cmd_query(args: argparse.Namespace) -> None:
    results = search(
        args.index_dir,
        args.q,
        embedder_name=args.embedder,
        top_k=args.top_k,
        threshold=args.threshold
    )
    if not results:
        print("No results found.")
        return
    print("Top results (with distances):")
    chunks = []
    for chunk, distance in results:
        chunks.append(chunk)
        print(f"  - (distance={distance:.4f}) {chunk.path}:{chunk.start_line}-{chunk.end_line} ({chunk.symbol})")
    context = assemble_context(chunks, max_chars=args.max_chars)
    print("\n----- CONTEXT -----\n")
    print(context)


def cmd_chat(args: argparse.Namespace) -> None:
    # Lazy import to avoid requiring 'requests' just to run --help
    from .llm_client import LLMClient
    prompt = build_prompt(
        index_dir=args.index_dir,
        user_query=args.q,
        embedder_name=args.embedder,
        top_k=args.top_k,
        max_context_chars=args.max_chars,
        threshold=args.threshold,
    )
    if args.dry_run:
        print("\n----- PROMPT (dry-run) -----\n")
        print(prompt)
        return
    client = LLMClient(api_url=args.api_url, api_key=args.api_key, model=args.model)
    out = client.complete(prompt, max_tokens=args.max_tokens, temperature=args.temperature)
    print(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codexbot", description="Code-aware retrieval + chat MVP")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="Index a repository")
    p_idx.add_argument("--repo", required=True, help="Path to repository root")
    p_idx.add_argument("--index-dir", default=".codexbot/index", help="Where to store the index")
    p_idx.add_argument("--embedder", default="hash", help="hash or sentence-transformers name/path")
    p_idx.set_defaults(func=cmd_index)

    p_q = sub.add_parser("query", help="Search and show context")
    p_q.add_argument("--index-dir", default=".codexbot/index")
    p_q.add_argument("--q", required=True, help="Natural language question or keywords")
    p_q.add_argument("--top-k", type=int, default=10)
    p_q.add_argument("--max-chars", type=int, default=12000)
    p_q.add_argument("--embedder", default=None)
    p_q.add_argument("--threshold", type=float, default=1.5, help="Similarity threshold")
    p_q.set_defaults(func=cmd_query)

    p_c = sub.add_parser("chat", help="Build prompt and call intranet LLM")
    p_c.add_argument("--index-dir", default=".codexbot/index")
    p_c.add_argument("--q", required=True)
    p_c.add_argument("--top-k", type=int, default=10)
    p_c.add_argument("--max-chars", type=int, default=12000)
    p_c.add_argument("--embedder", default=None)
    p_c.add_argument("--threshold", type=float, default=1.5, help="Similarity threshold")
    p_c.add_argument("--api-url", default=os.getenv("LLM_API_URL", ""))
    p_c.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    p_c.add_argument("--model", default=os.getenv("LLM_MODEL", ""))
    p_c.add_argument("--max-tokens", type=int, default=1024)
    p_c.add_argument("--temperature", type=float, default=0.2)
    p_c.add_argument("--dry-run", action="store_true", help="Print prompt instead of calling LLM")
    p_c.set_defaults(func=cmd_chat)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
