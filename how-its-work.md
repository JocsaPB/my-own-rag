# How It Works — RAG Local + Claude Code

## Por que BGE-M3?

Foram avaliados dois modelos para este RAG:

| Modelo | Dims | Context | Foco | Status |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384D | 256 tokens | Geral | Substituído |
| `jinaai/jina-embeddings-v2-base-code` | 768D | 8192 tokens | Código | Incompatível com transformers ≥4.40 |
| `BAAI/bge-m3` | 1024D | 8192 tokens | Multilingual | **Escolhido** |

O `jina-embeddings-v2-base-code` seria ideal para código puro, mas seu código customizado referencia `find_pruneable_heads_and_indices`, função removida do `transformers` na versão 4.40 e o modelo não foi atualizado. O BGE-M3 funciona com transformers 5.x, tem 1024D (representação mais rica) e contexto de 8192 tokens.

---

## O que é o BGE-M3

O `BAAI/bge-m3` (Beijing Academy of AI) é um modelo de **sentence embeddings** de última geração. Ele converte qualquer texto em um vetor de **1024 dimensões** que representa o significado semântico do conteúdo.

Textos semanticamente similares ficam próximos no espaço vetorial:

```
"função que valida e-mail"    → [0.04, -0.21, 0.67, ...]  (1024 números)
"verificar formato de email"  → [0.05, -0.20, 0.65, ...]  (muito próximo!)
"conexão com banco de dados"  → [0.81,  0.33, -0.12, ...]  (distante)
```

**Vantagens sobre all-MiniLM-L6-v2:**
- **8192 tokens** de contexto (era 256) — funções inteiras cabem em um chunk
- **1024 dimensões** (era 384) — representação muito mais rica e precisa
- **State-of-the-art** no MTEB benchmark (maior benchmark de retrieval)
- **Chunks maiores**: 6000 chars/chunk (era 2400) — menos fragmentação de código

---

## Visão Geral da Solução

```mermaid
flowchart TD
    subgraph SETUP["Fase 1 — Setup (uma vez)"]
        A[rag-setup.run] --> B[Instala dependências Python em ~/.rag_venv]
        B --> C[Sobe ChromaDB Docker localhost:8000]
        C --> D[Instala mcp-rag-server em ~/.local/bin/]
        D --> E[Configura ~/.claude.json com mcpServers]
    end

    subgraph INDEX["Fase 2 — Indexação"]
        F[indexer_full.py /projeto] --> G[Varre arquivos .py .js .ts .md ...]
        G --> H[Divide em chunks 6000 chars, overlap 800]
        H --> I[BAAI/bge-m3 encode chunks → vetores 1024D]
        I --> J[(ChromaDB coleção: codebase HNSW cosine index)]
    end

    subgraph QUERY["Fase 3 — Busca em tempo real"]
        K[Claude Code CLI] -->|stdio| L[mcp-rag-server]
        L --> M[BAAI/bge-m3 encode query → vetor 1024D]
        M --> N{ChromaDB HNSW search cosine distance}
        N --> O[Top-K chunks mais similares]
        O --> L
        L -->|resultados formatados| K
        K --> P[Claude usa o contexto para responder]
    end

    SETUP --> INDEX
    INDEX --> QUERY
```

---

## Diagrama de Sequencia — Busca Semantica

```mermaid
sequenceDiagram
    actor User as Usuario
    participant CC as Claude Code CLI
    participant MCP as mcp-rag-server<br/>(stdio)
    participant Model as BAAI/bge-m3<br/>(em memoria)
    participant DB as ChromaDB<br/>(Docker :8000)

    User->>CC: Pergunta sobre o codigo
    CC->>CC: Decide chamar semantic_search_code
    CC->>MCP: {tool: "semantic_search_code",<br/>query: "funcao de autenticacao", top_k: 7}

    MCP->>Model: encode(["funcao de autenticacao"])
    Model-->>MCP: vetor [0.04, -0.21, ...] (1024D)

    MCP->>DB: POST /api/v2/collections/codebase/query<br/>{query_embeddings: [...], n_results: 7}
    DB->>DB: HNSW search<br/>cosine distance entre<br/>query e todos os chunks
    DB-->>MCP: [{doc, metadata, distance}, ...]

    MCP->>MCP: Formata resultado<br/>similarity = (1 - distance) * 100

    MCP-->>CC: "## [1] /projeto/src/auth.py\nSimilaridade: 91.2%\n..."
    CC->>CC: Injeta resultado no contexto
    CC-->>User: Resposta usando o codigo encontrado
```

---

## Diagrama de Sequencia — Indexacao

```mermaid
sequenceDiagram
    actor Dev as Desenvolvedor
    participant IDX as indexer_full.py
    participant FS as Sistema de Arquivos
    participant Model as BAAI/bge-m3
    participant DB as ChromaDB

    Dev->>IDX: python3 indexer_full.py /meu-projeto

    IDX->>DB: heartbeat() — valida conexao
    DB-->>IDX: OK

    IDX->>Model: SentenceTransformer("BAAI/bge-m3", device="cpu")
    Model-->>IDX: modelo carregado (~570MB)

    IDX->>DB: get_or_create_collection("codebase", hnsw:space=cosine)
    DB-->>IDX: collection handle

    IDX->>FS: os.walk(root_path)
    FS-->>IDX: lista de arquivos (ignora .git, node_modules, binarios...)

    loop Para cada arquivo
        IDX->>FS: read_text(encoding=utf-8)
        FS-->>IDX: conteudo do arquivo

        IDX->>IDX: RecursiveCharacterTextSplitter<br/>chunk_size=6000, overlap=800<br/>separators=["\n\n", "\n", " ", ""]

        IDX->>Model: encode(chunks) → batch
        Model-->>IDX: [[...1024 floats...], ...]

        IDX->>IDX: make_chunk_id = MD5(file_path::chunk::index)

        IDX->>DB: collection.upsert(<br/>ids=[md5...], embeddings=[...],<br/>documents=[chunks], metadatas=[...]<br/>)
        DB-->>IDX: OK
    end

    IDX-->>Dev: "Indexacao concluida! N arquivos, M chunks"
```

---

## Como o Modelo Funciona Internamente

O `BGE-M3` e baseado em arquitetura XLM-RoBERTa com camadas de attention empilhadas:

```
Texto de entrada
      |
[Tokenizacao SentencePiece — multilingual]
      |
[N x Transformer Layers com Multi-Head Self-Attention]
      |
[Mean Pooling sobre tokens nao-padding]
      |
[Normalizacao L2]
      |
Vetor de 1024 dimensoes (embedding normalizado)
```

**Por que distancia cosseno?**

A distancia cosseno mede o angulo entre dois vetores, ignorando a magnitude. Ideal para embeddings normalizados porque:
- Textos curtos e longos sobre o mesmo topico ficam proximos
- A normalizacao L2 do BGE-M3 torna cosine = dot product (mais rapido)

```
similarity = 1 - cosine_distance
           = dot(A, B)   # quando ambos sao vetores L2-normalizados
```

---

## Fluxo de Dados Detalhado

```mermaid
flowchart LR
    subgraph INPUT["Entrada"]
        Q["Query do usuario
        'funcao de login'"]
    end

    subgraph EMBED["Embedding BGE-M3"]
        T1["Tokenizacao SentencePiece
        tokens multilinguals"]
        T2["Transformer Layers
        Multi-Head Attention"]
        T3["Mean Pooling + L2 Norm
        → vetor 1024D"]
        T1 --> T2 --> T3
    end

    subgraph SEARCH["Busca Vetorial HNSW"]
        V1["Vetor da query
        1024 dimensoes"]
        V2["HNSW Index
        ~N vetores indexados"]
        V3["Top-K vizinhos
        por cosine similarity"]
        V1 --> V2 --> V3
    end

    subgraph OUTPUT["Saida"]
        R1["[1] auth.py — 91%
        def login(user, pwd)..."]
        R2["[2] session.py — 87%
        def create_session()..."]
        R3["[3] middleware.py — 82%
        def require_auth()..."]
    end

    INPUT --> EMBED
    EMBED --> SEARCH
    SEARCH --> OUTPUT
```

---

## Estrutura dos Chunks no ChromaDB

Cada chunk armazenado tem:

| Campo | Tipo | Exemplo |
|---|---|---|
| `id` | string | `MD5("src/auth.py::chunk::0")` |
| `embedding` | float[1024] | `[0.04, -0.21, ...]` |
| `document` | string | `"def login(user, pwd):\n    ..."` |
| `file_path` | metadata | `/home/jocsa/projeto/src/auth.py` |
| `chunk_index` | metadata | `0` |
| `file_name` | metadata | `auth.py` |
| `relative_path` | metadata | `src/auth.py` |

O ID e deterministico: o mesmo arquivo + mesmo indice sempre gera o mesmo MD5. Isso permite **upsert idempotente** — reindexar um arquivo nao cria duplicatas.

---

## Parametros e seus Impactos

| Parametro | Valor anterior | Valor atual | Impacto |
|---|---|---|---|
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | BAAI/bge-m3 | Melhor qualidade, mais dimensoes |
| `CHUNK_SIZE` | 2400 chars | 6000 chars | Funcoes inteiras cabem em 1 chunk |
| `CHUNK_OVERLAP` | 400 chars | 800 chars | Mais continuidade entre chunks |
| `embedding dims` | 384D | 1024D | Representacao muito mais rica |
| `context window` | 256 tokens | 8192 tokens | Nao trunca mais codigo longo |
| `top_k` | 7 (padrao) | 7 (padrao) | Quantidade de chunks por busca |
| `MAX_FILE_SIZE` | 500KB | 500KB | Limite de tamanho de arquivo |
| `hnsw:space` | cosine | cosine | Melhor para embeddings normalizados |
| `device` | cpu | cpu | Sem dependencia de GPU |

---

## Integracao com Claude Code

O Claude Code se conecta ao `mcp-rag-server` via **stdio** (protocolo MCP). O servidor e iniciado como subprocesso quando o Claude Code carrega:

```
claude (processo principal)
    └── mcp-rag-server (subprocesso, stdio)
            ├── carrega BAAI/bge-m3 em memoria (~570MB)
            └── mantem conexao HTTP com ChromaDB
```

A comunicacao e JSON-RPC sobre stdin/stdout. O Claude envia uma chamada de ferramenta e recebe o resultado formatado de volta — tudo sem expor portas ou APIs externas.

---

## Migracao do modelo antigo

Ao trocar de modelo, a colecao ChromaDB precisa ser recriada — as dimensoes sao incompativeis (384D vs 1024D). O procedimento:

```bash
# 1. Deletar a colecao antiga
python3 -c "
import chromadb
c = chromadb.HttpClient(host='localhost', port=8000)
c.delete_collection('codebase')
print('Colecao deletada')
"

# 2. Reindexar o projeto com o novo modelo
python3 indexer_full.py /caminho/do/projeto
```
