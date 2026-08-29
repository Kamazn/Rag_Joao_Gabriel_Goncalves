from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel

from .chunking import create_chunks
from .config import DEFAULT_TOP_K
from .documents import (
    find_markdown_files,
    load_markdown_documents,
)
from .indexing import (
    create_vector_store,
    get_indexed_chunk_count,
    index_chunks,
)
from .retrieval import SearchResult, search_similar_chunks

@dataclass(frozen=True)
class IndexingSummary:
    file_count: int
    chunk_count: int
    indexed_count: int

@dataclass(frozen=True)
class RAGResponse:
    question: str
    results: list[SearchResult]
    answer: str | None

# Liga leitura chunking embedding e Chroma na etapa de indexacao
def run_indexing_pipeline(
    vector_store: Chroma | None = None,
) -> tuple[Chroma, IndexingSummary]:
    markdown_files = find_markdown_files()
    documents = load_markdown_documents(markdown_files)
    chunks = create_chunks(documents)

    store = index_chunks(
        chunks=chunks,
        vector_store=vector_store,
        reset_collection=True,
        show_progress=True,
    )

    summary = IndexingSummary(
        file_count=len(markdown_files),
        chunk_count=len(chunks),
        indexed_count=get_indexed_chunk_count(store),
    )

    return store, summary

# Liga a pergunta busca semantica e geracao da resposta
def run_query_pipeline(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    use_generation: bool = True,
    vector_store: Chroma | None = None,
    model: BaseChatModel | None = None,
    show_progress: bool = False,
) -> RAGResponse:
    store = vector_store or create_vector_store()

    if show_progress:
        print("Buscando os trechos mais relevantes...", flush=True)

    results = search_similar_chunks(
        query=query,
        vector_store=store,
        top_k=top_k,
    )

    answer: str | None = None

    if use_generation:
        # Importa o modelo de geracao somente quando ele vai ser usado
        from .generation import generate_answer

        if show_progress:
            print("Gerando a resposta com o Qwen...", flush=True)

        answer = generate_answer(
            query=query,
            results=results,
            model=model,
        )

    return RAGResponse(
        question=query,
        results=results,
        answer=answer,
    )