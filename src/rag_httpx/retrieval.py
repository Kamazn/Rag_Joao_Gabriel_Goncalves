from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K
from .indexing import get_indexed_chunk_count

# Junta o trecho encontrado com a similaridade e a posicao dele
@dataclass(frozen=True)
class SearchResult:
    document: Document
    similarity: float
    rank: int

def validate_query(query: str) -> str:
    clean_query = query.strip()

    if not clean_query:
        raise ValueError("A pergunta não pode estar vazia")

    return clean_query

def validate_top_k(top_k: int) -> None:
    if not MIN_TOP_K <= top_k <= MAX_TOP_K:
        raise ValueError(
            f"top_k precisa estar entre {MIN_TOP_K} e {MAX_TOP_K}"
        )

# Transforma a pergunta em embedding e procura os chunks mais proximos
def search_similar_chunks(
    query: str,
    vector_store: Chroma,
    top_k: int = DEFAULT_TOP_K,
) -> list[SearchResult]:
    clean_query = validate_query(query)
    validate_top_k(top_k)

    indexed_count = get_indexed_chunk_count(vector_store)

    if indexed_count == 0:
        raise ValueError(
            "O índice vetorial está vazio, execute a indexação primeiro"
        )

    search_limit = min(top_k, indexed_count)

    results_with_distance = vector_store.similarity_search_with_score(
        query=clean_query,
        k=search_limit,
    )

    results: list[SearchResult] = []

    # No cosseno quanto menor a distancia maior a similaridade
    for rank, (document, distance) in enumerate(
        results_with_distance,
        start=1,
    ):
        similarity = 1.0 - float(distance)
        similarity = max(-1.0, min(1.0, similarity))

        results.append(
            SearchResult(
                document=document,
                similarity=similarity,
                rank=rank,
            )
        )

    return results