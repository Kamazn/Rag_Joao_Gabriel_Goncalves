from pathlib import Path

import pytest

from src.rag_httpx.config import EXPECTED_MARKDOWN_FILES
from src.rag_httpx.documents import find_markdown_files


def test_finds_official_markdown_files() -> None:
    markdown_files = find_markdown_files()

    assert len(markdown_files) == EXPECTED_MARKDOWN_FILES
    assert all(path.suffix == ".md" for path in markdown_files)

# Cria uma base pequena e temporaria pra testar a busca recursiva
def test_finds_markdown_files_inside_subfolders(tmp_path: Path) -> None:
    subfolder = tmp_path / "advanced"
    subfolder.mkdir()

    first_file = tmp_path / "index.md"
    second_file = subfolder / "authentication.md"
    ignored_file = subfolder / "notes.txt"

    first_file.write_text("# Index", encoding="utf-8")
    second_file.write_text("# Authentication", encoding="utf-8")
    ignored_file.write_text("Esse arquivo nao deve entrar", encoding="utf-8")

    markdown_files = find_markdown_files(tmp_path)

    assert markdown_files == sorted([first_file, second_file])

# Confirma se aparece um erro quando a pasta existe mas nao tem nenhum arquivo
def test_rejects_empty_corpus(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Nenhum arquivo Markdown"):
        find_markdown_files(tmp_path)

# Confirma se aparece um erro quando a pasta da documentacao nao foi encontrada
def test_rejects_missing_docs_folder(tmp_path: Path) -> None:
    missing_folder = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="não encontrada"):
        find_markdown_files(missing_folder)