import re
from pathlib import Path

from langchain_core.documents import Document

from .config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS


MARKDOWN_HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
FENCE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})")
WORD_PATTERN = re.compile(r"\S+")


def validate_chunk_settings(
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> None:
    if chunk_size_words <= 0:
        raise ValueError("O tamanho do chunk precisa ser maior que zero")

    if chunk_overlap_words < 0:
        raise ValueError("O overlap não pode ser negativo")

    if chunk_overlap_words >= chunk_size_words:
        raise ValueError("O overlap precisa ser menor que o tamanho do chunk")


def split_markdown_sections(
    text: str,
    default_title: str,
) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = default_title
    current_lines: list[str] = []
    current_fence: str | None = None

    for line in text.splitlines():
        fence_match = FENCE_PATTERN.match(line)

        # Controla os blocos de codigo pra nao confundir comentarios com secoes
        if fence_match:
            fence_character = fence_match.group(1)[0]

            if current_fence is None:
                current_fence = fence_character
            elif current_fence == fence_character:
                current_fence = None

            current_lines.append(line)
            continue

        heading_match = None

        if current_fence is None:
            heading_match = MARKDOWN_HEADING_PATTERN.match(line)

        # Quando encontra outro titulo guarda a secao anterior
        if heading_match:
            section_content = "\n".join(current_lines).strip()

            if section_content:
                sections.append((current_title, section_content))

            current_title = heading_match.group(1).strip()
            current_lines = []
            continue

        current_lines.append(line)

    final_content = "\n".join(current_lines).strip()

    if final_content:
        sections.append((current_title, final_content))

    return sections


def split_text_with_overlap(
    text: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> list[str]:
    validate_chunk_settings(chunk_size_words, chunk_overlap_words)

    word_matches = list(WORD_PATTERN.finditer(text))

    if not word_matches:
        return []

    chunks: list[str] = []
    step = chunk_size_words - chunk_overlap_words
    start_word = 0

    # Usa a posicao das palavras pra cortar sem destruir as quebras do Markdown
    while start_word < len(word_matches):
        end_word = min(start_word + chunk_size_words, len(word_matches))

        start_character = word_matches[start_word].start()
        end_character = word_matches[end_word - 1].end()

        chunk_text = text[start_character:end_character].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end_word == len(word_matches):
            break

        start_word += step

    return chunks


def create_chunks(
    documents: list[Document],
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    chunk_overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[Document]:
    validate_chunk_settings(chunk_size_words, chunk_overlap_words)

    if not documents:
        raise ValueError("Nenhum documento foi recebido para criar os chunks")

    chunks: list[Document] = []

    for document in documents:
        source = document.metadata.get("source")

        if not isinstance(source, str) or not source.strip():
            raise ValueError("O documento não possui uma fonte válida")

        if not document.page_content.strip():
            raise ValueError(f"O documento está vazio: {source}")

        title = str(
            document.metadata.get("title") or Path(source).stem
        ).strip()

        sections = split_markdown_sections(
            text=document.page_content,
            default_title=title,
        )

        if not sections:
            raise ValueError(f"O documento não gerou nenhuma seção: {source}")

        chunk_index = 1

        for section_title, section_content in sections:
            section_chunks = split_text_with_overlap(
                text=section_content,
                chunk_size_words=chunk_size_words,
                chunk_overlap_words=chunk_overlap_words,
            )

            for section_chunk in section_chunks:
                chunk_id = f"{source}::chunk-{chunk_index:03d}"

                # Repete o titulo da secao pra dar contexto ao embedding
                chunk_content = f"{section_title}\n\n{section_chunk}".strip()

                chunk_metadata = {
                    **document.metadata,
                    "section": section_title,
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                }

                chunks.append(
                    Document(
                        page_content=chunk_content,
                        metadata=chunk_metadata,
                    )
                )

                chunk_index += 1

    return chunks