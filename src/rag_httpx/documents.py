import re
from pathlib import Path

from langchain_core.documents import Document

from .config import DOCS_PATH, HTTPX_REPOSITORY_PATH


FENCED_CODE_PATTERN = re.compile(
    r"```.*?```|~~~.*?~~~",
    re.DOTALL,
)
MARKDOWN_H1_PATTERN = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
HTML_H1_PATTERN = re.compile(
    r"<h1[^>]*>\s*(.*?)\s*</h1>",
    re.IGNORECASE | re.DOTALL,
)


def find_markdown_files(docs_path: Path = DOCS_PATH) -> list[Path]:
    # Confere se a pasta da documentacao existe
    if not docs_path.is_dir():
        raise FileNotFoundError(
            f"Pasta da documentação não encontrada: {docs_path}\n"
            "Confira se o HTTPX foi clonado dentro de data/httpx."
        )

    # Procura os Markdown nas subpastas e organiza os caminhos
    markdown_files = sorted(
        path
        for path in docs_path.rglob("*.md")
        if path.is_file()
    )

    # Impede o projeto de continuar com a base vazia
    if not markdown_files:
        raise ValueError(
            f"Nenhum arquivo Markdown foi encontrado dentro de: {docs_path}"
        )

    return markdown_files


def extract_document_title(text: str, file_path: Path) -> str:
    # Remove os blocos de codigo pra nao confundir comentarios com titulos
    searchable_text = FENCED_CODE_PATTERN.sub("", text)

    markdown_title = MARKDOWN_H1_PATTERN.search(searchable_text)
    html_title = HTML_H1_PATTERN.search(searchable_text)

    title_matches = [
        match
        for match in (markdown_title, html_title)
        if match is not None
    ]

    if title_matches:
        first_title = min(title_matches, key=lambda match: match.start())
        return first_title.group(1).strip()

    # Se nao encontrar H1 usa o nome do arquivo como titulo
    return file_path.stem.replace("-", " ").replace("_", " ").title()


def load_markdown_documents(
    file_paths: list[Path],
    repository_path: Path = HTTPX_REPOSITORY_PATH,
) -> list[Document]:
    documents: list[Document] = []

    for file_path in file_paths:
        text = file_path.read_text(encoding="utf-8")

        if not text.strip():
            raise ValueError(f"O arquivo Markdown está vazio: {file_path}")

        source_path = file_path.relative_to(repository_path).as_posix()

        # Guarda o texto junto com a fonte e o titulo pra manter a origem
        document = Document(
            page_content=text,
            metadata={
                "source": source_path,
                "title": extract_document_title(text, file_path),
            },
        )
        documents.append(document)

    return documents