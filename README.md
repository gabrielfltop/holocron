# 🔮 Holocron

> Chatbot de perguntas e respostas sobre os 6 filmes da saga Star Wars (Episódios I–VI), com RAG, cache em 2 níveis, roteamento de modelo e tool-calling sobre dados estruturados.

**Live demo:** <https://holocron.streamlit.app/>

## Problem statement

Fãs de Star Wars frequentemente querem consultar detalhes específicos dos filmes — falas, personagens, planetas, naves — mas navegar por wikis e roteiros manualmente é lento e impreciso. O Holocron resolve isso com um chatbot que responde perguntas em linguagem natural citando as fontes exatas. RAG é a abordagem certa porque o corpus é grande demais para caber no contexto; o retrieval semântico garante que só os trechos relevantes entrem no prompt, e o tool-calling cobre os casos em que o dado certo (altura, massa, planeta natal) precisa vir de uma busca exata, não semântica.

## Arquitetura

```
flowchart LR
    USER([Usuário]) --> EXACT{Exact cache<br/>SHA256?}
    EXACT -->|hit| RESP[Resposta]
    EXACT -->|miss| SEM{Semantic cache<br/>cos sim >= 0.93?}
    SEM -->|hit| RESP
    SEM -->|miss| ROUTE[classify_complexity]
    ROUTE -->|keyword analítica| PREMIUM[gemini-2.5-pro]
    ROUTE -->|default| CHEAP[gemini-2.5-flash-lite]
    CHEAP --> TOOLS{Precisa de tool?}
    PREMIUM --> TOOLS
    TOOLS -->|get_character_data /<br/>list_sources / filter_by_source| RAG[(Chroma RAG)]
    TOOLS -->|não| RAG
    RAG --> RESP
```

    Loading

O roteamento é **puramente por palavra-chave**, não por tamanho de pergunta: se a query contém `explique`, `compare`, `analise` ou `projete`, vai para o modelo premium; qualquer outra query (curta ou longa) cai no modelo barato por padrão. A geração roda com `temperature=0.0` para respostas determinísticas.

## Corpus

| Fonte                  | Formato        | Conteúdo                                                  |
| ----------------------- | -------------- | ----------------------------------------------------------- |
| imsdb.com                | `.txt`       | Roteiros completos dos Episódios I–VI                       |
| SWAPI (swapi.py4e.com)   | `.txt`       | Personagens, planetas, naves, veículos, espécies e filmes   |
| Genérico (opcional)     | `.pdf`       | Qualquer PDF em `data/corpus/` — extraído página a página com `pypdf` |

O pipeline aceita os dois formatos ao mesmo tempo: `.txt` são divididos em blocos por `---` (um bloco por cena/registro), PDFs são lidos página a página. Todos os blocos/páginas passam pelo mesmo `RecursiveCharacterTextSplitter` (`chunk_size=800`, `chunk_overlap=100`, separadores `["\n\n", "\n", ". ", " ", ""]`) antes de ir para o Chroma.

## Cache em 2 níveis

- **`ExactCache`** — hash SHA256 da query. Pega replays idênticos sem gastar embedding nem chamada de LLM.
- **`SemanticCache`** — embedda a query e compara por cosseno contra queries anteriores; se a similaridade for `>= 0.93`, reaproveita a resposta. Pega paráfrases ("quem é Luke" vs "me fale sobre o Luke Skywalker").

Ambos usam `GEMINI_API_KEY` + `gemini-embedding-001` quando disponível, com fallback para `text-embedding-3-small` da OpenAI.

## Roteamento de modelo

`classify_complexity(query)` decide o modelo antes de qualquer chamada de LLM:

- Contém `explique`, `compare`, `analise` ou `projete` → modelo **premium** (`PREMIUM_MODEL`, default `gemini-2.5-pro`)
- Caso contrário → modelo **barato** (`CHEAP_MODEL`, default `gemini-2.5-flash-lite`), independente do tamanho da pergunta

## Tools (function calling)

O agente expõe 3 tools sobre o pipeline, registradas em `TOOL_REGISTRY` via `init_tools(pipeline)`:

| Tool                 | O que faz                                                                                          |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| `get_character_data` | Busca exata (não semântica) por nome dentro dos `.txt` da SWAPI — evita falsos positivos do retrieval em atributos como altura, massa, ano de nascimento. |
| `list_sources`        | Lista todos os arquivos indexados no Chroma com contagem de chunks por fonte.                          |
| `filter_by_source`    | Busca semântica restrita a um arquivo específico (`where={"source": ...}`), ex.: só no roteiro do Episódio IV. |

## Indexação

`python build_corpus.py` carrega o `.env`, monta o pipeline e indexa o que ainda não estiver no Chroma (`ingest_and_index` só roda se `collection.count() < total_esperado`, então rodar o script de novo é seguro/idempotente).

Detalhes da indexação (`RAGPipeline.ingest_and_index`):

- Lotes de `BATCH_SIZE=10` chunks, com `4.5s` de espera entre lotes (respeita rate limit do Gemini free tier);
- Até `3` tentativas por lote, com backoff de `9s` (`4.5s * 2`) entre tentativas;
- Se um lote falhar em todas as tentativas, a indexação **para** (não continua silenciosamente) — evita queimar cota em lotes que provavelmente vão falhar pelo mesmo motivo (ex.: cota diária esgotada). O script informa quantos chunks ficaram de fora e pode ser rodado de novo depois.

## Setup

```bash
# 1. Clone o repositório
git clone <seu-repo>
cd holocron

# 2. Dependências
uv venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync

# 3. API key
cp .env.example .env
# edite .env — Gemini free tier é o default (GEMINI_API_KEY); OpenAI é o fallback

# 4. Gerar corpus (idempotente — pode rodar de novo se parar no meio)
python build_corpus.py

# 5. Rodar local
streamlit run src/ui/streamlit_app.py
```

> O `.env.example` também lista uma opção de chave Anthropic para o `LLM_MODEL`, mas hoje **só Gemini e OpenAI estão de fato implementados** em `rag.py`, `cache.py` e `routing.py` (o cliente é sempre `OpenAI(...)` apontado pro endpoint OpenAI-compatible do Gemini, ou direto pra OpenAI). Se for usar Claude, é preciso adicionar esse branch.

## Observability

Por padrão, o projeto usa **structured logging** (`trace_id` por requisição, modelo, tokens, latência) via `src/observability/trace.py` — suficiente para acompanhar custo e latência localmente.

Para métricas mais completas (dashboard, P95 de latência, custo estimado por request), o projeto está pronto para integração opcional com [Langfuse](https://langfuse.com): basta instalar `langfuse`, preencher `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` no `.env` e decorar as funções de chamada de LLM com `@observe()`. Detalhes em [`docs/observability.md`](docs/observability.md).

## Testes

`tests/test_smoke.py` cobre o caminho feliz do pipeline: indexação (`collection.count() > 0`), `retrieve()` (retorna lista com `text`/`source`/`distance`) e `answer()` (retorna `dict` com `answer` e `sources`).

```bash
uv run pytest tests/test_smoke.py -v
```

Os testes são pulados automaticamente (`pytest.skip`) se não houver `GEMINI_API_KEY`/`OPENAI_API_KEY` no `.env` ou se `data/corpus/` não tiver pelo menos um `.pdf`.

## Design decisions

- **Corpus em `.txt` separados por categoria + suporte a `.pdf` genérico:** facilita o `filter_by_source` — o agente pode restringir a busca a roteiros ou à SWAPI dependendo da pergunta, e qualquer PDF extra cai automaticamente no mesmo pipeline de chunking.
- **`chunk_size=800, overlap=100`:** tamanho suficiente pra capturar diálogos completos nos roteiros sem perder contexto entre cenas.
- **Cache em 2 níveis (exato + semântico):** exact cache é essencially grátis (hash local); semantic cache evita reprocessar paráfrases, mas custa 1 chamada de embedding por miss.
- **Indexação fail-fast:** parar após 3 tentativas falhas por lote, em vez de seguir tentando os lotes seguintes, evita desperdiçar cota quando o motivo da falha é sistêmico (cota esgotada, chave inválida).
- **Roteamento por palavra-chave, não por tamanho:** perguntas analíticas ("compare", "explique") tendem a precisar de mais raciocínio independentemente de quantos caracteres têm; roteamento por tamanho penalizaria perguntas curtas mas complexas.
- **`get_character_data` busca direta no `.txt`:** dados estruturados da SWAPI (altura, massa, ano de nascimento) são recuperados por match de nome exato/parcial, evitando falsos positivos do retrieval semântico.
- **`temperature=0.0` na geração:** respostas determinísticas, mais fáceis de avaliar e reproduzir.

## Limitations

- Cobre apenas os Episódios I–VI. Séries como The Mandalorian, Andor, Ahsoka e material Legends fora desses filmes não estão no corpus.
- Free tier do Gemini limita a indexação (batches pequenos + delay entre eles), então a indexação inicial do corpus completo é lenta.
- Suporte a Anthropic está documentado no `.env.example` mas não está implementado no código do pipeline — hoje só Gemini/OpenAI funcionam de fato.
- `test_smoke.py` exige pelo menos 1 PDF em `data/corpus/` além dos `.txt` já versionados — sem isso os testes são pulados, não falham.

## Tech stack

- **LLM:** Gemini 2.5 Flash-Lite (default, via endpoint OpenAI-compatible do Gemini) / Gemini 2.5 Pro (queries analíticas) — com fallback para OpenAI se `GEMINI_API_KEY` não estiver setada
- **Embeddings:** `gemini-embedding-001` (default) / `text-embedding-3-small` (fallback OpenAI)
- **Vector store:** Chroma local (`PersistentClient`)
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`
- **Leitura de PDF:** `pypdf`
- **UI:** Streamlit
- **Observability:** structured logs com `trace_id` (padrão) + Langfuse opcional
- **Testes:** pytest
- **Deploy:** Streamlit Community Cloud

## Estrutura

```
holocron/
├── data/
│   ├── corpus/               # roteiros (.txt) + SWAPI (.txt) + PDFs opcionais
│   └── chroma/                # vector store (gitignored)
├── docs/
│   └── observability.md      # guia de logging estruturado + integração Langfuse
├── src/
│   ├── ui/streamlit_app.py
│   ├── pipeline/
│   │   ├── rag.py            # chunk, embed, index, retrieve, generate
│   │   ├── tools.py          # get_character_data, list_sources, filter_by_source
│   │   ├── cache.py          # ExactCache + SemanticCache
│   │   └── routing.py        # classify_complexity + make_client
│   └── observability/trace.py
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
├── build_corpus.py
├── pyproject.toml
├── .env.example
└── README.md
```

---

*Desenvolvido por Gabriel Farias — disciplina "Desenvolvendo Software com IA Generativa" (Mod4 PPI).*
