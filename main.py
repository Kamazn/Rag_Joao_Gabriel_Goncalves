import argparse
import time
from typing import Any

# Mostra os trechos mesmo quando a geracao estiver desligada
def print_search_results(response: Any) -> None:
    print("\nTrechos recuperados:")

    for result in response.results:
        metadata = result.document.metadata
        source = metadata.get("source", "Fonte desconhecida")
        section = metadata.get("section", "Sem seção")

        print(
            f"{result.rank}. {source} | "
            f"Seção: {section} | "
            f"Similaridade: {result.similarity:.4f}"
        )

    if response.answer is not None:
        print("\nResposta:")
        print(response.answer)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG local da documentação do HTTPX"
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    commands.add_parser(
        "index",
        help="Lê os documentos e salva os embeddings no Chroma",
    )

    query_parser = commands.add_parser(
        "query",
        help="Faz uma pergunta sobre a documentação",
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

def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Importa as bibliotecas pesadas so depois de entender o comando
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
        print("\nExecução interrompida antes de terminar")
        return 130

    except Exception as error:
        print(f"\nErro: {error}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())