import argparse
import time
from typing import Any

# Mostra o texto e a origem de cada trecho recuperado
def print_search_results(response: Any) -> None:
    print("\nTrechos recuperados:")

    for result in response.results:
        metadata = result.document.metadata
        source = metadata.get("source", "Fonte desconhecida")
        section = metadata.get("section", "Sem seção")

        print(
            f"\n{result.rank}. {source} | "
            f"Seção: {section} | "
            f"Similaridade: {result.similarity:.4f}"
        )
        print("\nTrecho:")
        print(result.document.page_content)
        print("-" * 80)

    if response.answer is not None:
        print("\nResposta:")
        print(response.answer)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG local da documentação do HTTPX"
    )

    commands = parser.add_subparsers(
        dest="command",
    )

    commands.add_parser(
        "index",
        help="Lê os documentos e salva os embeddings no Chroma",
    )

    query_parser = commands.add_parser(
        "query",
        help="Executa uma pergunta diretamente pelo comando",
    )

    query_parser.add_argument(
        "question",
        help="Pergunta que será enviada ao RAG",
    )

    query_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        choices=range(3, 6),
        help="Quantidade de trechos retornados entre 3 e 5",
    )

    query_parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Retorna os trechos sem chamar o modelo de geração",
    )

    return parser

# Mantem o banco e o modelo carregados enquanto varias perguntas sao feitas
def run_interactive_chat(
    run_query_pipeline: Any,
) -> int:
    from src.rag_httpx.generation import create_generation_model
    from src.rag_httpx.indexing import (
        create_vector_store,
        get_indexed_chunk_count,
    )

    vector_store = create_vector_store()

    if get_indexed_chunk_count(vector_store) == 0:
        print(
            "\nO índice está vazio, execute primeiro:\n"
            "python main.py index"
        )
        return 1

    print("Preparando o Qwen...", flush=True)
    model = create_generation_model()

    print("\nRAG da documentação HTTPX")
    print("Digite sua pergunta normalmente")
    print("Digite 'sair' para encerrar")

    while True:
        try:
            question = input("\nPergunta: ").strip()
        except EOFError:
            print("\nEncerrando o RAG")
            return 0

        if question.casefold() in {"sair", "exit", "quit"}:
            print("Encerrando o RAG")
            return 0

        if not question:
            print("A pergunta não pode estar vazia")
            continue

        try:
            response = run_query_pipeline(
                query=question,
                vector_store=vector_store,
                model=model,
                show_progress=True,
            )

            print_search_results(response)

        except Exception as error:
            print(f"\nErro ao responder: {error}")

def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    try:
        print("Carregando o banco vetorial e os modelos locais...", flush=True)
        loading_start = time.perf_counter()

        from src.rag_httpx.pipeline import (
            run_indexing_pipeline,
            run_query_pipeline,
        )

        loading_time = time.perf_counter() - loading_start
        print(
            f"Dependências carregadas em {loading_time:.1f} segundos",
            flush=True,
        )

        # Sem comando abre o modo de perguntas interativas
        if args.command is None:
            return run_interactive_chat(run_query_pipeline)

        if args.command == "index":
            print("Iniciando a indexação no Chroma...", flush=True)

            _, summary = run_indexing_pipeline()

            print("\nIndexação concluída")
            print(f"Arquivos encontrados: {summary.file_count}")
            print(f"Chunks criados: {summary.chunk_count}")
            print(f"Chunks indexados: {summary.indexed_count}")

            return 0

        response = run_query_pipeline(
            query=args.question,
            top_k=args.top_k,
            use_generation=not args.retrieval_only,
            show_progress=True,
        )

        print_search_results(response)
        return 0

    except KeyboardInterrupt:
        print("\nExecução interrompida")
        return 130

    except Exception as error:
        print(f"\nErro: {error}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())