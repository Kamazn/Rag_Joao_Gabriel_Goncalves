from pathlib import Path

from .config import DOCS_PATH

 # Confere se a pasta da documentacao existe
def find_markdown_files(docs_path: Path = DOCS_PATH) -> list[Path]:
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