from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

from llm import MissingAPIKeyError

load_dotenv()


# RAG storage directory
RAG_DIR = Path(os.getenv("RAG_DIR", "./.rag"))
RAG_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = RAG_DIR / "rag_chunks.json"
INDEX_PATH = RAG_DIR / "rag_index.faiss"


EMBEDDING_MODEL = "text-embedding-3-small"
MOCK_EMBEDDING_DIM = 256
REAL_EMBEDDING_DIM = 1536


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_pdf_pages(path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(path)
    pages: List[Dict[str, Any]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = _normalize_whitespace(page.extract_text() or "")
        if not text:
            continue
        pages.append({"loc": page_index, "text": text})
    return pages


def extract_text_from_pdf(path: str) -> str:
    pages = _extract_pdf_pages(path)
    return "\n\n".join(page["text"] for page in pages)


def _chunk_text_word_based(text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    step = chunk_tokens - overlap_tokens
    if step <= 0:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    chunks: List[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_tokens]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words).strip())
        if start + chunk_tokens >= len(words):
            break
    return chunks


def chunk_text(text: str, chunk_tokens: int = 500, overlap_tokens: int = 50) -> List[str]:
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return []

    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(cleaned)
        if not tokens:
            return []

        step = chunk_tokens - overlap_tokens
        chunks: List[str] = []
        for start in range(0, len(tokens), step):
            token_slice = tokens[start : start + chunk_tokens]
            if not token_slice:
                continue
            chunk = encoding.decode(token_slice).strip()
            if chunk:
                chunks.append(chunk)
            if start + chunk_tokens >= len(tokens):
                break
        return chunks
    except Exception:
        return _chunk_text_word_based(cleaned, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)


def chunk_pdf(path: str, chunk_tokens: int = 500, overlap_tokens: int = 50) -> List[Dict[str, Any]]:
    source = Path(path).name
    records: List[Dict[str, Any]] = []
    for page in _extract_pdf_pages(path):
        loc = int(page["loc"])
        for chunk in chunk_text(str(page["text"]), chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens):
            records.append(
                {
                    "text": chunk,
                    "source": source,
                    "loc": loc,
                }
            )
    return records


def _use_mock_mode() -> bool:
    return os.getenv("MOCK_OPENAI", "false").strip().lower() in {"1", "true", "yes"}


def _current_embedding_dim() -> int:
    return MOCK_EMBEDDING_DIM if _use_mock_mode() else REAL_EMBEDDING_DIM


def _mock_embed_text(text: str, dim: int = MOCK_EMBEDDING_DIM) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim, dtype=np.float32)


def _embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.array([], dtype=np.float32)

    if _use_mock_mode():
        vectors = [_mock_embed_text(text) for text in texts]
        embeddings = np.vstack(vectors).astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key.strip():
        raise MissingAPIKeyError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    vectors: List[List[float]] = []
    batch_size = 128
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)

    embeddings = np.array(vectors, dtype="float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def _coerce_chunk_record(item: Any, fallback_chunk_id: int) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        text = _normalize_whitespace(item)
        if not text:
            return None
        return {
            "chunk_id": fallback_chunk_id,
            "text": text,
            "source": "unknown",
            "loc": None,
        }

    if not isinstance(item, dict):
        return None

    text = _normalize_whitespace(str(item.get("text", "")))
    if not text:
        return None

    raw_chunk_id = item.get("chunk_id", fallback_chunk_id)
    try:
        chunk_id = int(raw_chunk_id)
    except (TypeError, ValueError):
        chunk_id = fallback_chunk_id

    source = str(item.get("source", "unknown")).strip() or "unknown"
    raw_loc = item.get("loc")
    if raw_loc is None:
        loc = None
    else:
        try:
            loc = int(raw_loc)
        except (TypeError, ValueError):
            loc = None

    return {
        "chunk_id": chunk_id,
        "text": text,
        "source": source,
        "loc": loc,
    }


def _load_chunks() -> List[Dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        return []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return []

    chunks: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        coerced = _coerce_chunk_record(item, fallback_chunk_id=idx)
        if coerced:
            chunks.append(coerced)

    # Keep chunk IDs contiguous and deterministic.
    for idx, chunk in enumerate(chunks):
        chunk["chunk_id"] = idx

    return chunks


def _save_chunks(chunks: List[Dict[str, Any]]) -> None:
    for idx, chunk in enumerate(chunks):
        chunk["chunk_id"] = idx

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def _load_index() -> faiss.Index:
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    raise FileNotFoundError("rag_index.faiss not found. Run /ingest first.")


def _dedupe_key(record: Dict[str, Any]) -> tuple:
    return (
        str(record.get("text", "")),
        str(record.get("source", "unknown")),
        record.get("loc"),
    )


def build_or_update_index(
    texts: List[str],
    source: Optional[str] = None,
    locs: Optional[List[Optional[int]]] = None,
) -> int:
    incoming_records: List[Dict[str, Any]] = []
    default_source = source or "unknown"
    for idx, text in enumerate(texts):
        clean_text = _normalize_whitespace(text)
        if not clean_text:
            continue

        loc: Optional[int] = None
        if locs is not None and idx < len(locs):
            value = locs[idx]
            loc = int(value) if value is not None else None

        incoming_records.append(
            {
                "text": clean_text,
                "source": default_source,
                "loc": loc,
            }
        )

    return build_or_update_index_records(incoming_records)


def build_or_update_index_records(records: List[Dict[str, Any]], replace: bool = False) -> int:
    cleaned_records: List[Dict[str, Any]] = []
    for item in records:
        text = _normalize_whitespace(str(item.get("text", "")))
        if not text:
            continue
        source = str(item.get("source", "unknown")).strip() or "unknown"
        raw_loc = item.get("loc")
        if raw_loc is None:
            loc = None
        else:
            try:
                loc = int(raw_loc)
            except (TypeError, ValueError):
                loc = None
        cleaned_records.append({"text": text, "source": source, "loc": loc})

    if not cleaned_records:
        return 0

    existing_chunks = [] if replace else _load_chunks()
    force_rebuild = replace
    if not replace and INDEX_PATH.exists() and existing_chunks:
        existing_index = faiss.read_index(str(INDEX_PATH))
        if existing_index.d != _current_embedding_dim():
            force_rebuild = True
            existing_chunks = []

    existing_keys = {_dedupe_key(chunk) for chunk in existing_chunks}
    if force_rebuild:
        new_records = cleaned_records
    else:
        new_records = [record for record in cleaned_records if _dedupe_key(record) not in existing_keys]

    if not new_records:
        return 0

    new_embeddings = _embed_texts([record["text"] for record in new_records])

    if INDEX_PATH.exists() and existing_chunks and not replace:
        index = faiss.read_index(str(INDEX_PATH))
        if index.d != int(new_embeddings.shape[1]):
            index = faiss.IndexFlatIP(int(new_embeddings.shape[1]))
            existing_chunks = []
    else:
        index = faiss.IndexFlatIP(int(new_embeddings.shape[1]))
        existing_chunks = []

    start_chunk_id = len(existing_chunks)
    new_chunk_records: List[Dict[str, Any]] = []
    for offset, record in enumerate(new_records):
        new_chunk_records.append(
            {
                "chunk_id": start_chunk_id + offset,
                "text": record["text"],
                "source": record["source"],
                "loc": record["loc"],
            }
        )

    index.add(new_embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    all_chunks = existing_chunks + new_chunk_records
    _save_chunks(all_chunks)
    return len(new_chunk_records)


def retrieve(query: str, k: int = 5) -> List[dict]:
    chunks = _load_chunks()
    if not chunks or not INDEX_PATH.exists():
        return []

    query_clean = _normalize_whitespace(query)
    if not query_clean:
        return []

    index = _load_index()
    query_embedding = _embed_texts([query_clean])
    if query_embedding.size == 0:
        return []
    if index.d != int(query_embedding.shape[1]):
        return []

    top_k = min(max(k, 1), len(chunks))
    scores, indices = index.search(query_embedding, top_k)

    results: List[dict] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append(
            {
                "chunk_id": int(chunk["chunk_id"]),
                "text": str(chunk["text"]),
                "source": str(chunk.get("source", "unknown")),
                "loc": chunk.get("loc"),
                "score": float(score),
            }
        )
    return results


def get_chunk(chunk_id: int) -> Optional[Dict[str, Any]]:
    chunks = _load_chunks()
    if chunk_id < 0 or chunk_id >= len(chunks):
        return None
    chunk = chunks[chunk_id]
    return {
        "chunk_id": int(chunk["chunk_id"]),
        "source": str(chunk.get("source", "unknown")),
        "loc": chunk.get("loc"),
        "text": str(chunk.get("text", "")),
    }
