# RAG semântico para a documentação do HTTPX

Sistema local de recuperação e geração de respostas fundamentadas na documentação oficial do HTTPX.

## Identificação

- Nome do aluno: João Gabriel Gonçalves Batista
- Formato da solução: Script Python pelo terminal e interface web local com Streamlit
- Link do vídeo: PENDENTE
- Link do Colab, se aplicável: Não se aplica

## Objetivo

O sistema recebe perguntas sobre a documentação do HTTPX e retorna os trechos mais relevantes com suas fontes. Os documentos são divididos em chunks, transformados em embeddings e armazenados no Chroma. Depois da recuperação, o Qwen pode gerar uma resposta em português usando somente os trechos encontrados.

## Arquitetura resumida

```text
Indexação:

HTTPX/docs/**/*.md
        ↓
Leitura dos documentos
        ↓
Separação por seções Markdown
        ↓
Chunks com overlap e metadados
        ↓
Qwen3 Embedding
        ↓
Chroma com distância de cosseno

Consulta:

Pergunta pelo terminal ou Streamlit
        ↓
Embedding da pergunta
        ↓
Busca por similaridade no Chroma
        ↓
Top 3 a 5 chunks ordenados
        ↓
Trechos + fontes + seções + scores
        ↓
Qwen3.5 para geração opcional
```

## Como executar do zero

O projeto foi desenvolvido e testado com Python 3.14.3. Para executá-lo é necessário ter Git, Python e Ollama instalados.

### 1. Instale os programas necessários

#### Windows

Instale:

- Python: https://www.python.org/downloads/
- Git: https://git-scm.com/download/win
- Ollama: https://ollama.com/download/windows

O Ollama requer Windows 10 ou uma versão mais recente.

#### Linux

Instale o Python 3, o Git e o suporte para ambientes virtuais usando o gerenciador de pacotes da sua distribuição.

Instale o Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### macOS

Instale:

- Python: https://www.python.org/downloads/macos/
- Git: https://git-scm.com/download/mac
- Ollama: https://ollama.com/download/mac

O Ollama requer macOS 14 Sonoma ou uma versão mais recente.

### 2. Clone o projeto

Abra um terminal e execute:

```bash
git clone https://github.com/Kamazn/Rag_Joao_Gabriel_Goncalves.git
cd Rag_Joao_Gabriel_Goncalves
```

Os próximos comandos devem ser executados dentro da pasta do projeto.

### 3. Crie e ative o ambiente virtual

#### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

#### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Quando o ambiente estiver ativo, o terminal mostrará `(.venv)` antes do caminho atual.

### 4. Instale as dependências

Com o ambiente virtual ativo, execute:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Baixe a documentação do HTTPX

Clone o repositório oficial dentro de `data/httpx`:

```bash
git clone https://github.com/encode/httpx.git data/httpx
```

Fixe a versão exigida pelo desafio:

```bash
git -C data/httpx checkout b5addb64f0161ff6bfe94c124ef76f6a1fba5254
```

Confira a versão utilizada:

```bash
git -C data/httpx rev-parse HEAD
```

O resultado deve ser:

```text
b5addb64f0161ff6bfe94c124ef76f6a1fba5254
```

Nesse commit, o sistema deve encontrar 23 arquivos Markdown recursivamente dentro de `data/httpx/docs/`.

### 6. Prepare os modelos locais

No Windows e macOS, abra o aplicativo Ollama depois da instalação.

Confira se o Ollama está funcionando:

```bash
ollama list
```

Baixe o modelo de embeddings:

```bash
ollama pull qwen3-embedding:4b
```

Baixe o modelo de geração:

```bash
ollama pull qwen3.5:4b
```

Confira novamente:

```bash
ollama list
```

Os modelos `qwen3-embedding:4b` e `qwen3.5:4b` devem aparecer na lista.

Caso o serviço não esteja iniciado, execute em outro terminal:

```bash
ollama serve
```

### 7. Crie o índice vetorial

Com o ambiente virtual ativo e dentro da pasta do projeto, execute:

```bash
python main.py index
```

Esse comando:

1. encontra os 23 documentos Markdown;
2. lê os documentos e preserva seus metadados;
3. divide o conteúdo em chunks;
4. gera os embeddings com o Qwen;
5. salva o índice vetorial dentro de `data/chroma/`.

A primeira indexação pode demorar alguns minutos. O índice permanece salvo no Chroma e não precisa ser recriado antes de cada pergunta.

### 8. Abra o modo interativo

Execute:

```bash
python main.py
```

Quando o terminal mostrar:

```text
Pergunta:
```

Digite uma pergunta normalmente:

```text
Como configurar timeouts no HTTPX?
```

O sistema mostrará os trechos recuperados, suas fontes, seções, posições no ranking, scores de similaridade e a resposta gerada pelo Qwen.

Depois da resposta, outra pergunta poderá ser digitada. Para encerrar, digite:

```text
sair
```

### 9. Abra a interface web

A interface utiliza o mesmo pipeline do terminal. Execute:

```bash
python -m streamlit run app.py
```

O navegador deverá abrir automaticamente em `http://localhost:8501`.

Digite uma pergunta no campo localizado na parte inferior da página e pressione `Enter`. A interface mostra a resposta do Qwen e os chunks recuperados com fonte, seção, ranking, similaridade e conteúdo completo em caixas expansíveis.

O valor de `top_k` pode ser alterado entre 3 e 5 pela barra lateral. O padrão continua sendo 3. Para encerrar a página, pressione `Ctrl + C` no terminal.

### 10. Execute somente a recuperação

A geração é opcional. Para testar apenas a busca semântica, sem chamar o modelo gerador, execute:

```bash
python main.py query "Como configurar timeouts no HTTPX?" --top-k 3 --retrieval-only
```

Para comparar cinco resultados:

```bash
python main.py query "Como configurar timeouts no HTTPX?" --top-k 5 --retrieval-only
```

### 11. Execute os testes automatizados

Com o ambiente virtual ativo, execute:

```bash
python -m pytest -q
```

Resultado esperado:

```text
20 passed
```

## Decisões técnicas

### Chunking

- Estratégia: divisão por seções Markdown seguida por tamanho em palavras
- Tamanho aproximado: 60 palavras
- Overlap, se houver: 10 palavras
- Justificativa: a separação por seções evita misturar assuntos diferentes e o overlap preserva parte do contexto entre chunks vizinhos

Foram comparadas as configurações `60/10`, `80/15` e `90/15` usando oito perguntas com fontes esperadas. Todas encontraram a fonte correta no top 3 nas oito perguntas. A configuração `60/10` foi escolhida porque obteve o melhor MRR, com `0.8750`, enquanto as outras duas obtiveram `0.8125`.

A comparação pode ser repetida com `python evaluate_retrieval.py`. O script usa o modelo de embeddings local, cria índices temporários e não substitui o índice principal.

Os blocos de código são preservados para que comentários iniciados por `#` não sejam confundidos com títulos Markdown. O título da seção é repetido no conteúdo do chunk para ajudar o embedding a identificar o assunto.

### Embeddings e busca

- Modelo ou técnica: `qwen3-embedding:4b` executado localmente pelo Ollama
- Forma de cálculo da similaridade: distância de cosseno pelo Chroma
- Valor de `top_k`: 3 por padrão com valores permitidos entre 3 e 5
- Justificativa: o modelo consegue comparar perguntas em português com documentos em inglês e o Chroma mantém o índice salvo entre execuções

O mesmo modelo de embedding é utilizado nos documentos e nas perguntas. O score é usado para ordenar os resultados e não é tratado como uma probabilidade de acerto.

### Metadados e fontes

Cada chunk preserva o caminho do arquivo original, o título, a seção, a posição dentro do documento e um identificador único.

Os metadados utilizados são:

```text
source
title
section
chunk_index
chunk_id
```

Exemplo de identificador:

```text
docs/advanced/timeouts.md::chunk-001
```

Depois da busca o texto e esses metadados retornam juntos. Por isso a saída consegue mostrar o trecho, a fonte, a seção, a posição no ranking e a similaridade.

## Perguntas de teste

### 1. Pergunta com resposta clara

- Pergunta: Como configurar e desativar timeouts no HTTPX?
- Resultado esperado: Encontrar os trechos específicos sobre configuração e desativação de timeouts
- O resultado foi relevante? Por quê? Sim. O primeiro e o terceiro resultados vieram de `docs/advanced/timeouts.md` e o segundo veio de `docs/quickstart.md`. A resposta explicou o uso de `timeout=10.0`, `timeout=None` e o limite padrão de 5 segundos de inatividade

### 2. Pergunta ampla ou ambígua

- Pergunta: What features does HTTPX provide?
- Resultado esperado: Encontrar uma apresentação geral das funcionalidades do HTTPX
- O resultado foi relevante? Por quê? Parcialmente. `docs/index.md` apareceu no primeiro resultado com similaridade `0.8548`, mas os outros dois resultados apresentaram extensões de terceiros. A pergunta é ampla e não informa qual tipo de funcionalidade procura

### 3. Pergunta fora do escopo

- Pergunta: Qual é a capital da França?
- Como o sistema reagiu: O Chroma retornou três chunks com scores baixos entre `0.2311` e `0.2600`. O Qwen respondeu que a informação não estava presente porque os documentos tratam da documentação do HTTPX
- Como essa reação poderia melhorar: Poderia ser criado um limiar de similaridade calibrado com perguntas de teste para rejeitar automaticamente resultados muito fracos

## Limitações conhecidas

- A busca sempre retorna a quantidade definida em `top_k` mesmo quando a pergunta está fora do assunto
- Perguntas muito amplas podem recuperar documentos apenas parcialmente relacionados
- O score de similaridade não representa uma probabilidade
- A geração local pode ser mais lenta em computadores que utilizam apenas CPU
- O índice precisa ser recriado quando os documentos ou as configurações de chunking mudam
- A estratégia atual utiliza busca semântica sem combinar uma busca lexical
- Ainda não foi definido um limiar de similaridade para rejeitar resultados fracos

## Uso de ferramentas de IA

- Ferramentas utilizadas: Codex como apoio no desenvolvimento e Qwen pelo Ollama como modelo local de embeddings e geração de respostas
- Tarefas em que ajudaram: explicação dos conceitos de RAG, comparação das tecnologias, planejamento da arquitetura, sugestões de código e testes, diagnóstico de erros e revisão da documentação
- Exemplo representativo de prompt ou orientação: Ajude a desenvolver e entender um RAG em Python que encontre arquivos Markdown recursivamente, preserve a origem e a seção dos chunks, utilize embeddings locais, armazene os vetores no Chroma e permita testar cada etapa
- O que foi testado, modificado ou validado por você: defini a arquitetura do projeto e, após comparar diferentes opções, escolhi Python, Ollama, Qwen e Chroma; decidi usar execução local, busca semântica e `top_k` igual a 3; comparei três configurações de chunking e escolhi chunks de 60 palavras com overlap de 10 após os testes; implementei, revisei e adaptei as sugestões de código, organizei o projeto em módulos e analisei os resultados e os problemas encontrados durante os testes; validei a descoberta dos 23 documentos, títulos, seções, blocos de código, chunking, overlap, metadados, embeddings, persistência no Chroma, buscas em português e inglês, geração de respostas com fontes, perguntas fora do escopo e 20 testes automatizados

Não inclua conversas completas, chaves ou dados pessoais.

## Referências e código externo

- Repositório oficial do HTTPX: https://github.com/encode/httpx
- Ollama: https://ollama.com
- ChromaDB: https://docs.trychroma.com
- LangChain: https://python.langchain.com
- Pytest: https://docs.pytest.org
- Streamlit: https://docs.streamlit.io
- Vídeo utilizado para estudar RAG: https://www.youtube.com/watch?v=PZS9PS8rnc4
- Vídeo utilizado para estudar RAG e embeddings: https://www.youtube.com/watch?v=hgLpzI85-cs
- Vídeo utilizado como apoio durante o desenvolvimento: https://www.youtube.com/watch?v=yPz8LDcAcdA
- Vídeo utilizado para estudar avaliação e estratégias de RAG: https://www.youtube.com/watch?v=LZoLvV7p25A

## Segurança

Confirme uma opção:

- [x] Minha solução não usa API key.
- [ ] Minha solução usa segredo protegido e nenhuma chave foi publicada.

# Checklist final do aluno

## Conteúdo

- [x] A solução é individual.
- [x] O notebook ou código-fonte está incluído.
- [x] O `README.md` está preenchido.
- [x] As dependências e os passos de execução estão descritos.
- [x] O código usa `httpx/docs/**/*.md` no commit solicitado.
- [x] A saída mostra trechos e fontes.
- [x] Três tipos de pergunta foram testados.
- [x] Há reflexão sobre pelo menos uma limitação.

## Evidência

- [ ] O link do vídeo está no README e abre sem solicitar acesso adicional; ou
- [ ] O `DIAGNOSTICO.md` foi incluído porque a execução não foi concluída.

## Segurança e privacidade

- [x] Não há API keys, tokens ou senhas no código, notebook, histórico, vídeo ou ZIP.
- [ ] Notificações e dados pessoais não aparecem no vídeo.
- [ ] Links compartilhados concedem somente o acesso necessário.

## Envio

- [ ] O repositório abre para o professor ou o ZIP foi testado após ser descompactado.
- [x] O nome da entrega identifica o aluno.
- [ ] O envio foi concluído dentro da janela de sexta-feira, 28 de agosto de 2026, às 13h, até domingo, 30 de agosto de 2026, às 13h, no horário de Recife.

Abra sua própria entrega uma última vez em uma janela anônima ou ambiente limpo antes de enviá-la.
