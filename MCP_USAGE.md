# MCP RAG Server — Como Usar

Este documento explica como usar o MCP Server RAG para buscar código, atualizar índices e trabalhar com a codebase de forma semântica.

## Setup Rápido

O servidor MCP já deve estar configurado em `~/.claude.json`. Se não estiver:

```json
"mcpServers": {
  "rag-codebase": {
    "command": "~/.local/bin/mcp-rag-server",
    "args": [],
    "env": {
      "CHROMA_HOST": "localhost",
      "CHROMA_PORT": "8000"
    }
  }
}
```

**Pré-requisitos:**
- Docker rodando (`docker ps` deve mostrar o container chromadb)
- ChromaDB inicializado com dados indexados (`~/.rag_db`)
- mcp-rag-server instalado em `~/.local/bin/`

---

## Ferramentas Disponíveis

### 1. **semantic_search_code** — Busca Semântica

Encontra trechos de código relevantes usando busca vetorial. Funciona com descrições em linguagem natural.

**Parâmetros:**
- `query` (string): O que você está procurando — pode ser uma descrição vaga
- `top_k` (int, opcional): Quantos resultados retornar (padrão: 7, máximo: 20)

**Exemplos de uso:**

```
semantic_search_code("função que valida email")
semantic_search_code("como fazer requisição HTTP com retry", top_k=5)
semantic_search_code("integração com banco de dados", top_k=10)
```

**Resultado:**
```
# Resultados para: 'função que valida email'

## [1] /home/<usuario>/projeto/src/validators.py
**Similaridade:** 92.3%

```
def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[1]
```
```

**Dicas:**
- Use descrições em português ou inglês (o modelo entende bem)
- Seja específico: "função de autenticação" é melhor que "função"
- Se poucos resultados úteis, tente reformular a query
- Resultados acima de 85% geralmente são muito relevantes

---

### 2. **update_file_index** — Atualizar Índice de Arquivo

Após editar um arquivo, use isto para manter o índice RAG sincronizado.

**Parâmetros:**
- `file_path` (string): Caminho absoluto ou relativo do arquivo

**Exemplos:**

```
update_file_index("/home/<usuario>/projeto/src/auth.py")
update_file_index("src/validators.py")
update_file_index("<PROJECT_ROOT>/bin/mcp_server.py")
```

**Resultado:**
```
Arquivo reindexado com sucesso.
  Arquivo  : /home/<usuario>/projeto/src/auth.py
  Chunks antigos removidos: 5
  Novos chunks inseridos  : 6
```

**Quando usar:**
- Depois de editar um arquivo no projeto
- Depois de criar um novo arquivo
- Quando a IA faz mudanças no código que deveriam ser searchable

**Nota:** O arquivo é dividido em chunks (~2400 caracteres cada). Alterações pequenas podem afetar múltiplos chunks.

---

### 3. **delete_file_index** — Remover Arquivo do Índice

Remove um arquivo completamente do banco de dados vetorial.

**Parâmetros:**
- `file_path` (string): Caminho absoluto ou relativo

**Exemplos:**

```
delete_file_index("/home/<usuario>/projeto/src/old_module.py")
delete_file_index("temp/debug.py")
```

**Quando usar:**
- Quando um arquivo é deletado do projeto
- Quando você quer excluir um arquivo dos resultados de busca

---

### 4. **index_specific_folder** — Reindexar Pasta

Indexa ou reindexar todos os arquivos de um diretório.

**Parâmetros:**
- `folder_path` (string): Caminho da pasta a indexar recursivamente

**Exemplos:**

```
index_specific_folder("/home/<usuario>/projeto/src")
index_specific_folder("./src/auth")
```

**Resultado:**
```
Indexação da pasta concluída.
  Pasta    : /home/<usuario>/projeto/src
  Arquivos processados: 12/12
  Total de chunks     : 145
```

**Quando usar:**
- Depois de criar vários arquivos novos em uma pasta
- Para reindexar uma seção do projeto após alterações em massa
- Quando o índice está desatualizado para um diretório

---

## Fluxo Típico de Uso

### Buscar código existente
```
1. semantic_search_code("descrição do que procura")
2. Analisa os resultados
3. Navega para os arquivos mencionados
```

### Editar código
```
1. Lê o arquivo (Read tool)
2. Edita o arquivo (Edit tool)
3. update_file_index(file_path) — IMPORTANTE!
4. Próximas buscas já veem a versão nova
```

### Criar novo arquivo
```
1. Escreve o arquivo (Write tool)
2. index_specific_folder(folder_path) OU update_file_index(file_path)
3. Agora está pronto para ser encontrado em buscas
```

---

## Entendendo os Resultados

Cada resultado de busca inclui:

- **Número e rank** (ex: [1], [2], etc.)
- **Arquivo** — caminho completo para localizar o código
- **Similaridade %** — confiança da correspondência (70-100%)
- **Snippet** — até 800 caracteres do trecho mais relevante

**Como interpretar similaridade:**
- 90-100% → Muito relevante, provavelmente exatamente o que procura
- 80-89%  → Bastante relevante, confira o contexto completo
- 70-79%  → Relevante, mas pode precisar revisar mais contexto
- <70%   → Pode ser relevante apenas parcialmente

---

## Limitações e Comportamento

**O que o RAG sabe:**
- Código-fonte de linguagens de programação
- Documentação e markdown
- Configurações (JSON, YAML, etc.)
- Comentários no código

**O que ignora automaticamente:**
- Binários (`*.exe`, `*.so`, `*.dll`)
- Imagens (`*.png`, `*.jpg`)
- Mídia (`*.mp4`, `*.mp3`)
- Pacotes (`node_modules/`, `venv/`, `.git/`)
- Arquivos muito grandes (>500KB)

**Precisão:**
- O RAG é baseado em similaridade vetorial, não busca literal
- Reformule a query se os resultados não forem úteis
- A primeira chamada aquece o modelo (pode levar 1-2s)

---

## Troubleshooting

### "Erro de conexão: Não foi possível conectar ao ChromaDB"
```bash
# Verifique se Docker está rodando
docker ps | grep chromadb

# Se não estiver, inicie
docker compose -f "$HOME/docker-chromadb/docker-compose.yml" up -d
```

### "Nenhum resultado encontrado"
- O índice pode estar vazio — rode `python3 indexer_full.py /seu/projeto`
- Tente uma query mais simples
- Verifique se os arquivos estão indexados com `./chroma_monitor.sh chunks`

### Busca retorna resultados muito antigos
- Arquivo pode ter sido editado mas não atualizado no índice
- Use `update_file_index(file_path)` para sincronizar
- Ou `index_specific_folder(folder_path)` para toda a pasta

### Arquivo editado não aparece nos resultados
1. Confirme que o arquivo foi salvo
2. Use `update_file_index(file_path)` imediatamente após editar
3. Aguarde a próxima busca (o modelo está aquecido)

---

## Configuração Avançada

Todas as configurações são definidas em `~/.rag_venv/bin/mcp_server.py`:

```python
CHUNK_SIZE = 2400        # Caracteres por chunk (~600 tokens)
CHUNK_OVERLAP = 400      # Sobreposição entre chunks
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"  # Modelo de embeddings
TOP_K_RESULTS = 7        # Resultados padrão por busca
MAX_FILE_SIZE = 500KB    # Limite de tamanho de arquivo
```

Para alterar:
1. Edite `~/.rag_venv/bin/mcp_server.py`
2. Reinicie o servidor (reinicie o Claude Code)
3. Reindexe os dados: `python3 indexer_full.py /seu/projeto`

---

## Performance

- **Primeira busca:** 1-2 segundos (aquecimento do modelo)
- **Buscas subsequentes:** <1 segundo
- **Indexação de 1 arquivo:** <1 segundo
- **Indexação de pasta com 100 arquivos:** 10-30 segundos

---

## Exemplos Práticos

### Encontrar tratamento de erros
```
semantic_search_code("como fazer try catch ou exception handling")
```

### Localizar função de login
```
semantic_search_code("autenticação de usuário login", top_k=5)
```

### Buscar integração com API
```
semantic_search_code("chamada a API externa requests HTTP")
```

### Encontrar testes
```
semantic_search_code("testes unitários pytest")
```

### Procurar por padrões de cache
```
semantic_search_code("cache memoização performance")
```

---

## Integração com Claude Code

O MCP Server é automaticamente chamado pelo Claude Code quando você:

1. **Usa a ferramenta de busca** — procura por "semantic_search_code"
2. **Edita um arquivo** — sincroniza automaticamente com `update_file_index`
3. **Pede recomendações de código** — busca contexto relevante antes de responder

A IA pode usar todas as 4 ferramentas para:
- Entender o projeto antes de fazer mudanças
- Encontrar padrões existentes e seguir convenções
- Manter o índice atualizado enquanto trabalha
- Evitar duplicação de código ao sugerir novos

---

**Último update:** 2026-03-05 — Documentação para mcp-rag-server v1.0
