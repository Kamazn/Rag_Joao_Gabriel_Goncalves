from dataclasses import dataclass
from time import perf_counter

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.rag_httpx.chunking import create_chunks
from src.rag_httpx.documents import (
    find_markdown_files,
    load_markdown_documents,
)
from src.rag_httpx.indexing import (
    create_embedding_model,
    create_vector_store,
    index_chunks,
)
from src.rag_httpx.config import MIN_SIMILARITY
from src.rag_httpx.retrieval import search_similar_chunks

@dataclass(frozen=True)
class EvaluationQuestion:
    query: str
    expected_sources: tuple[str, ...]

@dataclass(frozen=True)
class EvaluationSummary:
    chunk_size: int
    overlap: int
    chunk_count: int
    hits: int
    mean_reciprocal_rank: float
    indexing_seconds: float

QUESTIONS = (
    EvaluationQuestion(
        query="Como configurar e desativar timeouts no HTTPX?",
        expected_sources=("docs/advanced/timeouts.md",),
    ),
    EvaluationQuestion(
        query="How do I send asynchronous requests with HTTPX?",
        expected_sources=("docs/async.md",),
    ),
    EvaluationQuestion(
        query="Como criar uma autenticação personalizada no HTTPX?",
        expected_sources=("docs/advanced/authentication.md",),
    ),
    EvaluationQuestion(
        query="Como habilitar o suporte a HTTP/2 no HTTPX?",
        expected_sources=("docs/http2.md",),
    ),
    EvaluationQuestion(
        query="How do I configure a proxy with HTTPX?",
        expected_sources=("docs/advanced/proxies.md",),
    ),
    EvaluationQuestion(
        query="Como configurar certificados SSL no HTTPX?",
        expected_sources=("docs/advanced/ssl.md",),
    ),
    EvaluationQuestion(
        query="How can I use event hooks in HTTPX?",
        expected_sources=("docs/advanced/event-hooks.md",),
    ),
    EvaluationQuestion(
        query="What features does HTTPX provide?",
        expected_sources=("docs/index.md", "docs/quickstart.md"),
    ),
)

# Perguntas sem relacao com o HTTPX, usadas pra calibrar o limiar de similaridade
OUT_OF_SCOPE_QUERIES = (
    "Qual é a capital da França?",
    "Como fazer um bolo de chocolate?",
    "Quem ganhou a Copa do Mundo de 2014?",
    "What is the best way to learn guitar?",
)

CONFIGURATIONS = (
    (60, 10),
    (80, 15),
    (90, 15),
)

# Testa se a fonte esperada aparece entre os tres primeiros resultados
def evaluate_configuration(
    documents: list[Document],
    embedding_model: Embeddings,
    chunk_size: int,
    overlap: int,
) -> EvaluationSummary:
    chunks = create_chunks(
        documents=documents,
        chunk_size_words=chunk_size,
        chunk_overlap_words=overlap,
    )

    collection_name = f"httpx_eval_{chunk_size}_{overlap}"
    vector_store = create_vector_store(
        embedding_model=embedding_model,
        persist_directory=None,
        collection_name=collection_name,
    )

    indexing_start = perf_counter()

    try:
        index_chunks(
            chunks=chunks,
            vector_store=vector_store,
            show_progress=True,
        )

        indexing_seconds = perf_counter() - indexing_start
        hits = 0
        reciprocal_rank_sum = 0.0
        in_scope_scores: list[float] = []

        print(
            f"\nConfiguração: chunk={chunk_size} overlap={overlap}"
        )

        for number, question in enumerate(QUESTIONS, start=1):
            results = search_similar_chunks(
                query=question.query,
                vector_store=vector_store,
                top_k=3,
            )

            expected_rank = next(
                (
                    result.rank
                    for result in results
                    if result.document.metadata.get("source")
                    in question.expected_sources
                ),
                None,
            )

            in_scope_scores.append(results[0].similarity)

            if expected_rank is not None:
                hits += 1
                reciprocal_rank_sum += 1.0 / expected_rank

            returned_sources = ", ".join(
                str(result.document.metadata.get("source"))
                for result in results
            )

            status = (
                f"acerto no rank {expected_rank}"
                if expected_rank is not None
                else "fonte esperada não apareceu"
            )

            print(f"{number}. {status} | {returned_sources}")

        # Compara o pior acerto com o melhor resultado fora do assunto
        out_of_scope_scores = [
            search_similar_chunks(
                query=query,
                vector_store=vector_store,
                top_k=3,
            )[0].similarity
            for query in OUT_OF_SCOPE_QUERIES
        ]

        print(
            "Calibração do limiar | "
            f"menor top 1 no assunto: {min(in_scope_scores):.4f} | "
            f"maior top 1 fora do assunto: {max(out_of_scope_scores):.4f} | "
            f"limiar atual: {MIN_SIMILARITY:.2f}"
        )

        return EvaluationSummary(
            chunk_size=chunk_size,
            overlap=overlap,
            chunk_count=len(chunks),
            hits=hits,
            mean_reciprocal_rank=reciprocal_rank_sum / len(QUESTIONS),
            indexing_seconds=indexing_seconds,
        )

    finally:
        vector_store.delete_collection()

def main() -> None:
    markdown_files = find_markdown_files()
    documents = load_markdown_documents(markdown_files)
    embedding_model = create_embedding_model()
    summaries: list[EvaluationSummary] = []

    for chunk_size, overlap in CONFIGURATIONS:
        summaries.append(
            evaluate_configuration(
                documents=documents,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    ordered_summaries = sorted(
        summaries,
        key=lambda summary: (
            summary.hits,
            summary.mean_reciprocal_rank,
        ),
        reverse=True,
    )

    print("\nResumo da avaliação")

    for summary in ordered_summaries:
        print(
            f"chunk={summary.chunk_size} "
            f"overlap={summary.overlap} "
            f"chunks={summary.chunk_count} "
            f"acertos={summary.hits}/{len(QUESTIONS)} "
            f"MRR={summary.mean_reciprocal_rank:.4f} "
            f"indexação={summary.indexing_seconds:.1f}s"
        )

    winner = ordered_summaries[0]

    print(
        "\nMelhor configuração: "
        f"chunk={winner.chunk_size} overlap={winner.overlap}"
    )

if __name__ == "__main__":
    main()
