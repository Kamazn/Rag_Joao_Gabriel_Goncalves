from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.rag_httpx.config import MIN_SIMILARITY, NOT_FOUND_MESSAGE
from src.rag_httpx.generation import NOT_FOUND_MARKER, generate_answer
from src.rag_httpx.indexing import create_vector_store, index_chunks
from src.rag_httpx.pipeline import run_query_pipeline
from src.rag_httpx.retrieval import (
    SearchResult,
    filter_relevant_results,
    validate_query,
)

SOURCE = "docs/advanced/timeouts.md"


# Cada assunto tem um eixo proprio, assuntos desconhecidos nao se parecem com nenhum documento
class TopicEmbeddings(Embeddings):
    topics = ("timeout", "http2", "authentication")

    def create_vector(self, text: str) -> list[float]:
        clean_text = text.lower()
        vector = [float(clean_text.count(topic)) for topic in self.topics]

        if not any(vector):
            return [0.0] * len(self.topics) + [1.0]

        return vector + [0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.create_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.create_vector(text)


@pytest.fixture
def vector_store():
    store = create_vector_store(
        embedding_model=TopicEmbeddings(),
        persist_directory=None,
        collection_name=f"test_scope_{uuid4().hex}",
    )

    chunks = [
        Document(
            page_content=f"HTTPX {topic} documentation",
            metadata={
                "chunk_id": f"{topic}-001",
                "chunk_index": 1,
                "source": f"docs/{topic}.md",
                "title": topic,
                "section": topic,
            },
        )
        for topic in TopicEmbeddings.topics
    ]

    index_chunks(chunks=chunks, vector_store=store)

    try:
        yield store
    finally:
        store.delete_collection()


def create_result(similarity: float) -> SearchResult:
    return SearchResult(
        document=Document(
            page_content="Example",
            metadata={"source": SOURCE},
        ),
        similarity=similarity,
        rank=1,
    )


# Confirma se so os trechos acima do limiar continuam no resultado
def test_filters_weak_results() -> None:
    results = [
        create_result(MIN_SIMILARITY + 0.2),
        create_result(MIN_SIMILARITY - 0.2),
    ]

    relevant = filter_relevant_results(results)

    assert relevant == [results[0]]


# Confirma se perguntas gigantes sao bloqueadas antes do embedding
def test_rejects_too_long_query() -> None:
    with pytest.raises(ValueError, match="no máximo"):
        validate_query("a" * 5000)


# Confirma se o modelo dizendo que nao achou nao devolve fontes
def test_not_found_answer_has_no_sources() -> None:
    fake_model = FakeListChatModel(responses=[NOT_FOUND_MARKER])

    answer = generate_answer(
        query="Qual é a capital da França?",
        results=[create_result(0.9)],
        model=fake_model,
    )

    assert answer == NOT_FOUND_MESSAGE
    assert SOURCE not in answer


# Confirma se uma pergunta fora do assunto nao vaza fontes nem trechos
def test_out_of_scope_query_does_not_leak_sources(vector_store) -> None:
    fake_model = FakeListChatModel(responses=["Resposta inventada"])

    response = run_query_pipeline(
        query="Qual é a capital da França?",
        vector_store=vector_store,
        model=fake_model,
    )

    assert response.found is False
    assert response.results == []
    assert response.answer == NOT_FOUND_MESSAGE
    assert "docs/" not in response.answer
    # O modelo nem chega a ser chamado quando nada e relevante
    assert fake_model.i == 0


# Confirma se uma pergunta do assunto continua com resposta e fonte
def test_in_scope_query_keeps_sources(vector_store) -> None:
    fake_model = FakeListChatModel(
        responses=["HTTPX permite configurar timeouts"]
    )

    response = run_query_pipeline(
        query="Como funciona o timeout?",
        vector_store=vector_store,
        model=fake_model,
    )

    assert response.found is True
    assert response.results[0].document.metadata["source"] == "docs/timeout.md"
    assert "docs/timeout.md" in response.answer
