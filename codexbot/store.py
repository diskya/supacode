from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import chromadb

@dataclass
class Chunk:
    id: str
    path: str
    start_line: int
    end_line: int
    symbol: str
    kind: str
    lang: str
    text: str

    @classmethod
    def from_chroma(cls, doc: Dict[str, Any]) -> "Chunk":
        return cls(
            id=doc['id'],
            path=doc['metadata']['path'],
            start_line=doc['metadata']['start_line'],
            end_line=doc['metadata']['end_line'],
            symbol=doc['metadata']['symbol'],
            kind=doc['metadata']['kind'],
            lang=doc['metadata']['lang'],
            text=doc['document']
        )

class ChromaStore:
    """
    Vector store using ChromaDB for persistence.
    """

    def __init__(self, index_dir: str) -> None:
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.index_dir)
        self.collection = self.client.get_or_create_collection(name="codex")

    def init_empty(self) -> None:
        # ChromaDB's get_or_create_collection handles initialization.
        # If we need to reset, we can delete and recreate the collection.
        try:
            self.client.delete_collection(name="codex")
        except ValueError:
            # Collection didn't exist, which is fine.
            pass
        self.collection = self.client.get_or_create_collection(name="codex")

    def add_chunks_and_vectors(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """Adds chunks and their corresponding vectors to the store."""
        if not chunks:
            return

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "path": c.path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "symbol": c.symbol,
                "kind": c.kind,
                "lang": c.lang,
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query_vector: List[float], limit: int = 5) -> List[Tuple[Chunk, float]]:
        """Performs a vector search and returns chunks with their distances."""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            include=["metadatas", "documents", "distances"]
        )
        
        if not results['ids'][0]:
            return []

        # The query returns lists of lists, one for each query vector. We only have one.
        docs_with_distances = []
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]
            
            chunk = Chunk(
                id=doc_id,
                path=metadata['path'],
                start_line=metadata['start_line'],
                end_line=metadata['end_line'],
                symbol=metadata['symbol'],
                kind=metadata['kind'],
                lang=metadata['lang'],
                text=document
            )
            docs_with_distances.append((chunk, distance))
            
        return docs_with_distances

    def get_chunks_by_ids(self, ids: List[str]) -> List[Chunk]:
        """Retrieves chunks by their IDs."""
        if not ids:
            return []
        
        results = self.collection.get(ids=ids)
        
        return [
            Chunk(
                id=results['ids'][i],
                path=results['metadatas'][i]['path'],
                start_line=results['metadatas'][i]['start_line'],
                end_line=results['metadatas'][i]['end_line'],
                symbol=results['metadatas'][i]['symbol'],
                kind=results['metadatas'][i]['kind'],
                lang=results['metadatas'][i]['lang'],
                text=results['documents'][i]
            )
            for i in range(len(results['ids']))
        ]
