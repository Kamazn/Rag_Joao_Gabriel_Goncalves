import pytest
from langchain_core.documents import Document

from src.rag_httpx.chunking import (
    create_chunks,
    split_text_with_overlap,
)

# Cria um documento pequeno pra testar sem depender da base do HTTPX
def create_test_document(text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "docs/example.md",
            "title": "Example",
        },
    )

# Confirma se os chunks sao criados com a fonte e a ordem de cada um
def test_creates_chunks_with_metadata() -> None:
    text = "# Installation\n" + " ".join(
        f"word{index}"
        for index in range(1, 11)
    )

    chunks = create_chunks(
        documents=[create_test_document(text)],
        chunk_size_words=6,
        chunk_overlap_words=2,
    )

    assert len(chunks) == 2
    assert chunks[0].metadata["source"] == "docs/example.md"
    assert chunks[0].metadata["section"] == "Installation"
    assert chunks[0].metadata["chunk_index"] == 1
    assert chunks[0].metadata["chunk_id"] == "docs/example.md::chunk-001"
    assert chunks[1].metadata["chunk_index"] == 2

# Confirma se o final de um chunk aparece no comeco do proximo
def test_keeps_overlap_between_chunks() -> None:
    text = "one two three four five six seven eight nine ten"

    chunks = split_text_with_overlap(
        text=text,
        chunk_size_words=6,
        chunk_overlap_words=2,
    )

    first_words = chunks[0].split()
    second_words = chunks[1].split()

    assert first_words[-2:] == second_words[:2]

# Confirma se os titulos do Markdown separam as secoes sem confundir o codigo
def test_respects_markdown_sections() -> None:
    text = "\n".join(
        [
            "# First section",
            "",
            "First content",
            "",
            "```shell",
            "# This is a command comment",
            "```",
            "",
            "## Second section",
            "",
            "Second content",
        ]
    )

    chunks = create_chunks(
        documents=[create_test_document(text)],
        chunk_size_words=50,
        chunk_overlap_words=5,
    )

    assert [chunk.metadata["section"] for chunk in chunks] == [
        "First section",
        "Second section",
    ]
    assert "# This is a command comment" in chunks[0].page_content

# Confirma se configuracoes que quebrariam o chunking sao bloqueadas
@pytest.mark.parametrize(
    ("chunk_size_words", "chunk_overlap_words"),
    [
        (0, 0),
        (10, -1),
        (10, 10),
    ],
)
def test_rejects_invalid_chunk_settings(
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> None:
    with pytest.raises(ValueError):
        create_chunks(
            documents=[create_test_document("Example content")],
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
        )

# Confirma se aparece um erro quando nao existe documento pra dividir
def test_rejects_empty_document_list() -> None:
    with pytest.raises(ValueError, match="Nenhum documento"):
        create_chunks([])