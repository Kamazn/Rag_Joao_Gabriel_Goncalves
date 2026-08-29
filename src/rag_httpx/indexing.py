from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from .config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    EMBEDDING_MODEL,
)

# Cria a conexao com o modelo que transforma os chunks em vetores
def create_embedding_model(
    model_name: str = EMBEDDING_MODEL,
) -> OllamaEmbeddings:
    try:
        return OllamaEmbeddings(
            model=model_name,
            validate_model_on_init=True,
        )
    except Exception as error:
        raise RuntimeError(
            f"Não foi possível carregar o modelo de embedding: {model_name}"
        ) from error

# Abre o Chroma local usando similaridade de cosseno
def create_vector_store(
    embedding_model: Embeddings | None = None,
    persist_directory: Path | str | None = CHROMA_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    directory: str | None = None

    if persist_directory is not None:
        directory_path = Path(persist_directory)
        directory_path.mkdir(parents=True, exist_ok=True)
        directory = str(directory_path)

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model or create_embedding_model(),
        persist_directory=directory,
        collection_configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )

# Confere os chunks e separa os ids que vao entrar no Chroma
def validate_chunks_for_indexing(
    chunks: list[Document],
) -> list[str]:
    if not chunks:
        raise ValueError("Nenhum chunk foi recebido para indexação")

    chunk_ids: list[str] = []
    seen_ids: set[str] = set()

    for chunk in chunks:
        if not chunk.page_content.strip():
            raise ValueError("Um chunk vazio foi recebido para indexação")

        chunk_id = chunk.metadata.get("chunk_id")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Um chunk não possui um chunk_id válido")

        if chunk_id in seen_ids:
            raise ValueError(f"chunk_id duplicado encontrado: {chunk_id}")

        seen_ids.add(chunk_id)
        chunk_ids.append(chunk_id)

    return chunk_ids

# Envia os chunks em grupos menores pra nao sobrecarregar o Ollama
def index_chunks(
    chunks: list[Document],
    vector_store: Chroma | None = None,
    reset_collection: bool = True,
    batch_size: int = 16,
    show_progress: bool = False,
) -> Chroma:
    if batch_size <= 0:
        raise ValueError("O tamanho do lote precisa ser maior que zero")

    chunk_ids = validate_chunks_for_indexing(chunks)
    store = vector_store or create_vector_store()

    if reset_collection:
        store.reset_collection()

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))

        store.add_documents(
            documents=chunks[start:end],
            ids=chunk_ids[start:end],
        )

        if show_progress:
            print(f"Chunks indexados: {end}/{len(chunks)}")

    return store

# Consulta somente os ids pra descobrir quantos chunks estao salvos
def get_indexed_chunk_count(vector_store: Chroma) -> int:
    stored_data = vector_store.get(include=[])
    return len(stored_data["ids"])