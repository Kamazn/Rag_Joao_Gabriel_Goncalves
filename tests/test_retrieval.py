from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.rag_httpx.indexing import (
    create_vector_store,
    get_indexed_chunk_count,
    index_chunks,
)
from src.rag_httpx.retrieval import search_similar_chunks

# Cria vetores simples pros testes nao dependerem do Ollama
class KeywordEmbeddings(Embeddings):
    keywords = ("http2", "timeout", "async", "authentication")

    def create_vector(self, text: str) -> list[float]:
        clean_text = text.lower()

        vector = [
            float(clean_text.count(keyword))
            for keyword in self.keywords
        ]

        vector.append(1.0)
        return vector

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [self.create_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.create_vector(text)

def create_test_chunk(
    chunk_id: str,
    content: str,
    source: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "chunk_index": 1,
            "source": source,
            "title": "HTTPX",
            "section": "Test",
        },
    )

# Confirma se o Chroma retorna primeiro o trecho mais parecido
def test_indexes_and_retrieves_similar_chunks() -> None:
    collection_name = f"test_httpx_{uuid4().hex}"

    vector_store = create_vector_store(
        embedding_model=KeywordEmbeddings(),
        persist_directory=None,
        collection_name=collection_name,
    )

    chunks = [
        create_test_chunk(
            chunk_id="timeout",
            content="HTTPX allows timeout configuration for requests",
            source="docs/advanced/timeouts.md",
        ),
        create_test_chunk(
            chunk_id="http2",
            content="HTTPX supports the HTTP2 protocol",
            source="docs/http2.md",
        ),
        create_test_chunk(
            chunk_id="authentication",
            content="HTTPX supports custom authentication",
            source="docs/advanced/authentication.md",
        ),
    ]

    try:
        index_chunks(
            chunks=chunks,
            vector_store=vector_store,
            batch_size=2,
        )

        results = search_similar_chunks(
            query="How do HTTPX timeouts work?",
            vector_store=vector_store,
            top_k=3,
        )

        assert get_indexed_chunk_count(vector_store) == 3
        assert len(results) == 3
        assert results[0].document.metadata["chunk_id"] == "timeout"
        assert results[0].rank == 1
        assert results[0].similarity >= results[1].similarity

    finally:
        vector_store.delete_collection()