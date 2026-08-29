import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.rag_httpx.generation import (
    build_context,
    generate_answer,
)
from src.rag_httpx.indexing import validate_chunks_for_indexing
from src.rag_httpx.retrieval import (
    SearchResult,
    validate_query,
    validate_top_k,
)

def create_result() -> SearchResult:
    document = Document(
        page_content="HTTPX supports configurable request timeouts",
        metadata={
            "source": "docs/advanced/timeouts.md",
            "title": "Timeouts",
            "section": "Setting a default timeout",
            "chunk_id": "timeout-001",
        },
    )

    return SearchResult(
        document=document,
        similarity=0.95,
        rank=1,
    )

# Confirma se perguntas vazias sao bloqueadas
def test_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="não pode estar vazia"):
        validate_query("   ")

# Confirma se o top k continua dentro da regra de 3 a 5
@pytest.mark.parametrize("top_k", [2, 6])
def test_rejects_invalid_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="entre 3 e 5"):
        validate_top_k(top_k)

# Confirma se uma lista vazia nao pode ser indexada
def test_rejects_empty_chunk_list() -> None:
    with pytest.raises(ValueError, match="Nenhum chunk"):
        validate_chunks_for_indexing([])

# Confirma se dois chunks nao entram no Chroma com o mesmo id
def test_rejects_duplicated_chunk_ids() -> None:
    chunk = Document(
        page_content="Example",
        metadata={
            "chunk_id": "duplicated",
        },
    )

    with pytest.raises(ValueError, match="duplicado"):
        validate_chunks_for_indexing([chunk, chunk])

# Confirma se a fonte e o trecho entram no contexto do prompt
def test_builds_context_with_source() -> None:
    context = build_context([create_result()])

    assert "docs/advanced/timeouts.md" in context
    assert "HTTPX supports configurable request timeouts" in context

# Testa a geracao sem precisar iniciar o Qwen de verdade
def test_generates_answer_with_fake_model() -> None:
    fake_model = FakeListChatModel(
        responses=[
            "HTTPX permite configurar timeouts"
        ]
    )

    answer = generate_answer(
        query="Como configurar timeouts?",
        results=[create_result()],
        model=fake_model,
    )

    assert "HTTPX permite configurar timeouts" in answer
    assert "docs/advanced/timeouts.md" in answer
