from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data"
HTTPX_REPOSITORY_PATH = DATA_PATH / "httpx"
DOCS_PATH = HTTPX_REPOSITORY_PATH / "docs"
CHROMA_PATH = DATA_PATH / "chroma"

# Versão exigida do httpx + o numero de markdowns esperado
HTTPX_COMMIT = "b5addb64f0161ff6bfe94c124ef76f6a1fba5254"
EXPECTED_MARKDOWN_FILES = 23

# Modelos que vao ser utilizados e idioma que a resposta vai sair
EMBEDDING_MODEL = "qwen3-embedding:4b"
GENERATION_MODEL = "qwen3.5:4b"
DEFAULT_RESPONSE_LANGUAGE = "pt-BR"

# Comeco com esses valores de chunking, depois ajusto usando os testes
CHUNK_SIZE_WORDS = 80
CHUNK_OVERLAP_WORDS = 15

DEFAULT_TOP_K = 3
MIN_TOP_K = 3
MAX_TOP_K = 5

CHROMA_COLLECTION_NAME = "httpx_docs"