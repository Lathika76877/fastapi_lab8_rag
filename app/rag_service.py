import io
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException, UploadFile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy.orm import Session
from . import models

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Initialize embedding model once at module load
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)


async def extract_text_from_upload(file: UploadFile) -> List[Tuple[str, Dict[str, Any]]]:
    filename = file.filename or "untitled.txt"
    ext = filename.lower().split(".")[-1]
    raw_bytes = await file.read()

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if ext in {"txt", "md"}:
        text_content = raw_bytes.decode("utf-8", errors="ignore")
        return [(text_content, {"source": filename, "file_type": ext, "page": 1})]

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(
                    (page_text, {"source": filename, "file_type": "pdf", "page": idx})
                )
        if not pages:
            raise HTTPException(
                status_code=400, detail="No extractable text found in PDF"
            )
        return pages

    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Please upload .txt, .md, or .pdf files.",
    )


async def ingest_document(
    db: Session, file: UploadFile, user_id: int
) -> List[models.DocumentChunk]:
    pages = await extract_text_from_upload(file)
    filename = file.filename or "untitled.txt"

    chunk_texts: List[str] = []
    chunk_metadatas: List[Dict[str, Any]] = []

    global_idx = 0
    for page_text, base_meta in pages:
        splits = text_splitter.split_text(page_text)
        for split in splits:
            if split.strip():
                meta = {
                    **base_meta,
                    "chunk_index": global_idx,
                    "char_count": len(split),
                }
                chunk_texts.append(split)
                chunk_metadatas.append(meta)
                global_idx += 1

    if not chunk_texts:
        raise HTTPException(
            status_code=400, detail="Document produced zero valid text chunks"
        )

    # Batch-generate 384-dim embeddings for all chunks
    vectors = embeddings_model.embed_documents(chunk_texts)

    # Idempotent re-ingestion: remove previous chunks for the same filename
    db.query(models.DocumentChunk).filter(
        models.DocumentChunk.source_filename == filename
    ).delete(synchronize_session=False)

    db_chunks = [
        models.DocumentChunk(
            source_filename=filename,
            chunk_index=meta["chunk_index"],
            content=text,
            metadata_json=meta,
            uploaded_by=user_id,
            embedding=vec,
        )
        for text, meta, vec in zip(chunk_texts, chunk_metadatas, vectors)
    ]

    db.add_all(db_chunks)
    db.commit()
    for chunk in db_chunks:
        db.refresh(chunk)

    return db_chunks


def search_similar_chunks(
    db: Session,
    query: str,
    top_k: int = 3,
    source_filename: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query_vector = embeddings_model.embed_query(query)
    distance_expr = models.DocumentChunk.embedding.cosine_distance(query_vector)

    stmt = db.query(models.DocumentChunk, distance_expr.label("distance"))
    if source_filename:
        stmt = stmt.filter(models.DocumentChunk.source_filename == source_filename)

    rows = stmt.order_by(distance_expr.asc()).limit(top_k).all()

    results = []
    for chunk, distance in rows:
        similarity = round(max(0.0, 1.0 - float(distance)), 4)
        results.append(
            {
                "id": chunk.id,
                "source_filename": chunk.source_filename,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "metadata_json": chunk.metadata_json,
                "similarity_score": similarity,
            }
        )
    return results