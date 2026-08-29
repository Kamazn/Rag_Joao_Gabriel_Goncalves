from typing import Any

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel

from src.rag_httpx.generation import (
    create_generation_model,
    generate_answer,
)
from src.rag_httpx.indexing import (
    create_vector_store,
    get_indexed_chunk_count,
)
from src.rag_httpx.pipeline import run_query_pipeline

st.set_page_config(
    page_title="RAG CHAT HTTPX",
    page_icon="☕",
    layout="centered",
)

# Mantem o Chroma carregado enquanto a pagina estiver aberta
@st.cache_resource(show_spinner=False, scope="session")
def load_vector_store() -> Chroma:
    return create_vector_store()

# Mantem o Qwen carregado pra nao iniciar o modelo em cada pergunta
@st.cache_resource(show_spinner=False, scope="session")
def load_generation_model() -> BaseChatModel:
    return create_generation_model()

# Transforma os resultados em dados simples pra guardar no historico
def prepare_results(results: list[Any]) -> list[dict[str, Any]]:
    prepared_results: list[dict[str, Any]] = []

    for result in results:
        metadata = result.document.metadata

        prepared_results.append(
            {
                "rank": result.rank,
                "source": metadata.get("source", "Fonte desconhecida"),
                "section": metadata.get("section", "Sem seção"),
                "similarity": result.similarity,
                "content": result.document.page_content,
            }
        )

    return prepared_results

# Mostra a resposta e deixa os chunks dentro de caixas expansivas
def show_assistant_message(message: dict[str, Any]) -> None:
    st.markdown(message["answer"])

    if message.get("generation_error"):
        st.warning(message["generation_error"])

    st.markdown("#### Trechos recuperados")

    for result in message["results"]:
        label = (
            f'{result["rank"]}. {result["source"]} '
            f'— {result["section"]}'
        )

        with st.expander(label):
            st.caption(
                f'Similaridade: {result["similarity"]:.4f}'
            )
            st.code(
                result["content"],
                language="markdown",
            )

st.title("RAG semântico para a documentação do HTTPX")
st.caption(
    "Faça perguntas sobre o HTTPX e receba respostas fundamentadas "
    "nos documentos oficiais"
)

with st.spinner("Carregando o índice vetorial..."):
    vector_store = load_vector_store()

indexed_count = get_indexed_chunk_count(vector_store)

if indexed_count == 0:
    st.error(
        "O índice vetorial está vazio. Execute o comando abaixo "
        "antes de abrir a página."
    )
    st.code("python main.py index", language="bash")
    st.stop()

with st.sidebar:
    st.header("Configuração")

    top_k = st.select_slider(
        "Quantidade de trechos",
        options=[3, 4, 5],
        value=3,
    )

    st.write(f"Chunks indexados: {indexed_count}")
    st.caption(
        "O valor padrão é 3 e os resultados são ordenados "
        "pela similaridade"
    )

    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            show_assistant_message(message)

question = st.chat_input(
    "Faça uma pergunta sobre a documentação do HTTPX"
)

if question:
    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando os trechos mais relevantes..."):
                retrieval_response = run_query_pipeline(
                    query=question,
                    top_k=top_k,
                    use_generation=False,
                    vector_store=vector_store,
                )

            generation_error: str | None = None

            try:
                with st.spinner("Gerando a resposta com o Qwen..."):
                    model = load_generation_model()

                    answer = generate_answer(
                        query=question,
                        results=retrieval_response.results,
                        model=model,
                    )

            except Exception as error:
                answer = (
                    "A recuperação dos documentos funcionou, mas não foi "
                    "possível gerar a resposta em linguagem natural."
                )
                generation_error = (
                    f"Erro durante a geração: {error}"
                )

            assistant_message = {
                "role": "assistant",
                "answer": answer,
                "results": prepare_results(
                    retrieval_response.results
                ),
                "generation_error": generation_error,
            }

            show_assistant_message(assistant_message)
            st.session_state.messages.append(assistant_message)

        except Exception as error:
            st.error(f"Não foi possível realizar a busca: {error}")