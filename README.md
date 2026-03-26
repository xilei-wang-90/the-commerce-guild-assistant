# the-commerce-guild-assistant
A RAG-powered knowledge assistant for the My Time series (Sandrock & Portia) using multi-representation indexing. Optimizes retrieval of complex wiki structures and tables for high-context LLMs.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Activating the environment

Before running any scripts, activate the Poetry virtual environment:

```bash
eval "$(poetry env activate)"
```

### Installing dependencies

```bash
poetry install
```

### Adding dependencies

```bash
poetry add <package-name>
```

## ChromaDB

The project uses [ChromaDB](https://www.trychroma.com/) as its vector store. ChromaDB runs as a local server.

### Starting ChromaDB

```bash
chroma run --path ./sandrock_db
```

This starts the ChromaDB server on `http://localhost:8000` with data persisted to `./sandrock_db/`.

### Verifying ChromaDB is running

```bash
curl http://localhost:8000/api/v2/heartbeat
```

A successful response returns a JSON object with a nanosecond timestamp, confirming the server is up.

## Running Tests

Run the test suite with pytest:

```bash
pytest
```

## Scripts

All scripts are run from the project root with the Poetry environment active.

### `scripts/run_scraper.py` — Scrape the wiki

Scrapes all pages from the My Time at Sandrock Fandom wiki and saves them as `.html` and `.md` files under `data/raw/`.

```bash
python3 scripts/run_scraper.py
```

Configuration constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `OUTPUT_DIR` | `data/raw/` | Directory where scraped files are saved |
| `MAX_PAGES` | `0` | Maximum pages to scrape; `0` means unlimited |
| `NUM_WORKERS` | `5` | Number of concurrent worker threads |

### `scripts/copy_to_silver.py` — Filter and promote to silver

Copies cleaned Markdown files from `data/raw/` to `data/silver/`, skipping meta/changelog pages. For buyback files, prepends a `# <filename>` heading.

```bash
python3 scripts/copy_to_silver.py
```

Requires `data/raw/` to exist (run the scraper first).

### `scripts/run_summarizer.py` — Generate LLM summaries

Reads all `.md` files from `data/silver/`, sends each to a local Ollama model, and writes concise summaries to `data/summaries/`.

Requires [Ollama](https://ollama.com/) to be running locally. The summarizer uses a custom Ollama model (`sandrock-model`) by default, which is defined in the `Modelfile` at the project root. Create it before running the summarizer:

```bash
ollama create sandrock-model -f Modelfile
```

Then run:

```bash
python3 scripts/run_summarizer.py
```

Configuration constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `INPUT_DIR` | `data/silver/` | Directory containing source Markdown files |
| `OUTPUT_DIR` | `data/summaries/` | Directory where summaries are written |
| `MODEL_NAME` | `sandrock-model` | Custom Ollama model to use for summarization (see `Modelfile`) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |

Existing summaries are skipped automatically. To regenerate all summaries, use the `--force` flag:

```bash
python3 scripts/run_summarizer.py --force
```

### `scripts/run_embedder.py` — Embed content into ChromaDB

Embeds `.md` files into `./sandrock_db` (ChromaDB) using the `all-minilm`
model served by a local Ollama instance. The embedding strategy is selected
via `--mode`:

| Mode | Source directory | Silver directory | Collection |
|---|---|---|---|
| `summary` | `data/summaries/` | `data/silver/` | `sandrock_wiki_summary` |
| `section-reverse-hyde` | `data/reverse-hyde/` | `data/silver-sections/` | `sandrock_wiki_section_reverse_hyde` |
| `section-tagged-reverse-hyde` | `data/tagged-reverse-hyde/` | `data/silver-sections/` | `sandrock_wiki_section_tagged_reverse_hyde` |

If `--mode` is omitted, the script prompts interactively.

Requires Ollama to be running locally with the model pulled:

```bash
ollama pull all-minilm
```

Each entry in the collection stores metadata including:

| Metadata key | Description |
|---|---|
| `filename` | e.g. `yakmel.md` |
| `summary_path` | Path to the embedded source file (summary or reverse-hyde questions) |
| `silver_path` | Path to the corresponding full-text file in the silver directory |

The `silver_path` is the key for downstream retrieval: when a similarity
match is found, use `silver_path` to load the full article text.

```bash
python3 scripts/run_embedder.py --mode summary
python3 scripts/run_embedder.py --mode section-reverse-hyde
python3 scripts/run_embedder.py --mode section-tagged-reverse-hyde
```

The database is persisted to `./sandrock_db/` automatically — no server
required. Existing entries are skipped on subsequent runs.

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--mode` | prompted | Embedding mode: `summary`, `section-reverse-hyde`, or `section-tagged-reverse-hyde` |
| `--force` | off | Re-embed files even if already in the database |
| `--max-records` | `10` | Maximum number of files to embed per run (`0` = unlimited) |
| `--ollama-url` | `http://localhost:11434` | Ollama server base URL |

```bash
python3 scripts/run_embedder.py --mode summary --force --max-records 0
```

### `scripts/drop_collection.py` — Drop a ChromaDB collection

Deletes a named collection and all its embeddings from the local ChromaDB
database. Irreversible without re-running the embedder.

```bash
python3 scripts/drop_collection.py
```

Defaults to the `sandrock_wiki_summary` collection. Use `--collection` to
target a different one:

```bash
python3 scripts/drop_collection.py --collection my_other_collection
```

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--collection` | `sandrock_wiki_summary` | Name of the collection to drop |
| `--db-path` | `./sandrock_db` | Path to the ChromaDB database directory |

Exits with an error if the collection does not exist.

### `scripts/run_chat.py` — Interactive knowledge assistant

Launches an interactive REPL chatbot. Each question is embedded, the top 5
candidates are retrieved from ChromaDB, then a cross-encoder reranker
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each candidate against
the actual silver-tier document content and selects the best 3. The
reranked results are used to build an augmented prompt for the LLM.

Requires `--mode` to select which collection to query:

- **`summary`** — queries `sandrock_wiki_summary` (full-text pages from `data/silver/`).
- **`section-reverse-hyde`** — queries `sandrock_wiki_section_reverse_hyde` (per-section chunks from `data/silver-sections/`).
- **`section-tagged-reverse-hyde`** — queries `sandrock_wiki_section_tagged_reverse_hyde` (per-section chunks with metadata tags from `data/silver-sections/`).

If `--mode` is omitted the script prompts interactively.

Requires Ollama running locally with `all-minilm` (embedding).
Needs `GEMINI_API_KEY` in the environment or `.env` for generation.

```bash
python3 scripts/run_chat.py --mode summary
python3 scripts/run_chat.py --mode section-reverse-hyde
python3 scripts/run_chat.py --mode section-tagged-reverse-hyde
```

| Flag | Default | Description |
|---|---|---|
| `--mode` | *(prompted)* | Retrieval mode: `summary`, `section-reverse-hyde`, or `section-tagged-reverse-hyde` |
| `--ollama-url` | `http://localhost:11434` | Ollama server base URL |

Configuration constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `DB_PATH` | `./sandrock_db` | Path to the ChromaDB database |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite-preview` | Cloud model for generation |
| `EMBEDDING_MODEL` | `all-minilm` | Ollama embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking |

Type `quit`, `exit`, or press Ctrl+C to stop.

### `scripts/run_web.py` — Launch Chainlit web UI

Starts a Chainlit web server that provides a browser-based chat interface to
the RAG pipeline. Users can switch between `summary`, `section-reverse-hyde`, and
`section-tagged-reverse-hyde` retrieval modes via the settings panel (gear
icon) in the UI.

Requires Ollama running locally with `all-minilm` (embedding) and
`GEMINI_API_KEY` in the environment or `.env`.

```bash
python3 scripts/run_web.py
python3 scripts/run_web.py --host 0.0.0.0 --port 8080
```

Or run directly via Chainlit:

```bash
chainlit run src/guild_assistant_web/app.py -w
```

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--host` | `localhost` | Host to bind to |
| `--port` | `8000` | Port to listen on |
| `-w` / `--watch` | off | Enable auto-reload on file changes |

### `scripts/run_section_breaker.py` — Break silver files into type-aware chunks

Reads all `.md` files from `data/silver/`, classifies each page by type,
and writes per-chunk files to `data/silver-sections/`.

**Page type classification** is based on filename and heading content:

| Type | Detection rule |
|---|---|
| Item | Contains a section called "Obtaining" |
| Location | Contains a section called "Region" |
| Character | Contains "Biographical Information" or "Physical Description" section |
| Monster | Contains a "Battle Statistics" section |
| Mission | Filename starts with `mission` |
| Dialogue | Filename ends with `dialogue` |
| Buyback | Filename ends with `buyback` |
| Event | Filename starts with `event` |
| Store | Contains a "Stock" section |
| Region | Contains a "Population" section |
| Festival | Contains a "Time" section |

**Chunking rules:**

- Each L2 section (with all L3+ subsections) forms one chunk
- Sections designated as "Overview" for the page type are merged into a
  single **Overview chunk** (`<page_slug>-overview.md`)
- Remaining L2 sections become standalone chunks (`<page_slug>-<section_slug>.md`)
- Chunks that contain only headings (no body text) are skipped
- Pre-heading content and L1 content always go to the Overview chunk
- "Overview", "Information", "General Information" sections always go to Overview
- If an "Overview" section is present, all sections before it also go to Overview

Type-specific overview sections:

| Type | Additional overview sections |
|---|---|
| Item | All sections before "Obtaining" |
| Location | "Establishment Information", "Locations" |
| Character | "Biographical Information", "Physical Description", "Personal Information", "Social Interactions", "Residence" |
| Monster | "Battle Statistics", "Drops" |
| Mission | "Mission Details", "Rewards" (first only), "Chronology" |
| Event | "Event Information", "Rewards" (first only) |
| Store | "Establishment Information" |
| Festival | "Time", "Information", "Unlock" |

```bash
python3 scripts/run_section_breaker.py
```

Configuration constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `INPUT_DIR` | `data/silver/` | Directory containing source Markdown files |
| `OUTPUT_DIR` | `data/silver-sections/` | Directory where chunk files are written |

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--force` | off | Re-break files even if output already exists |
| `--max-files` | `10` | Maximum number of source files to process per run (`0` = unlimited) |

```bash
python3 scripts/run_section_breaker.py --force --max-files 0
```

### `scripts/run_question_generator.py` — Generate hypothetical questions (Reverse HyDE)

Reads all `.md` files from `data/silver-sections/`, asks the LLM to list the
questions that each section answers, and writes the results to
`data/reverse-hyde/`.

This implements the **Reverse HyDE** indexing strategy: instead of embedding
raw text or summaries, you embed the questions a document answers, which
improves retrieval recall for conversational queries.

The prompt includes the **page title** and **section title** (derived from
the filename) to give the LLM context. Each section generates **1-3
questions**. The output contains questions only — no section headers, no
opening words, no numbering.

Requires [Ollama](https://ollama.com/) to be running locally. Large files are
automatically routed to Gemini (requires `GEMINI_API_KEY`).

```bash
python3 scripts/run_question_generator.py
```

Configuration constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `INPUT_DIR` | `data/silver-sections/` | Directory containing per-section Markdown files |
| `OUTPUT_DIR` | `data/reverse-hyde/` | Directory where question files are written |
| `MODEL_NAME` | `sandrock-model` | Local Ollama model to use |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Cloud model fallback for large files |

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--force` | off | Re-generate questions even if output already exists |
| `--max-files` | `10` | Maximum number of files to process per run (`0` = unlimited) |

```bash
python3 scripts/run_question_generator.py --force --max-files 0
```

### `scripts/run_question_tagger.py` — Tag reverse-hyde questions with metadata

Reads hypothetical question files from `data/reverse-hyde/` and the
corresponding silver-section Markdown files from `data/silver-sections/`.
For each file, extracts the page slug (from the filename) and all L2/L3
heading titles (from the section content), then writes a tagged copy to
`data/tagged-reverse-hyde/` with a bracketed tag line prepended.

No LLM calls are made — this is a pure text-processing step.

Tag format example:

```
[yakmel | Obtaining | From NPCs | From Enemies]
What items do yakmel drop?
How do you obtain a yakmel?
```

```bash
python3 scripts/run_question_tagger.py
```

Configuration constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `QUESTIONS_DIR` | `data/reverse-hyde/` | Directory containing reverse-hyde question files |
| `SECTIONS_DIR` | `data/silver-sections/` | Directory containing silver-section Markdown files |
| `OUTPUT_DIR` | `data/tagged-reverse-hyde/` | Directory where tagged question files are written |

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--force` | off | Re-tag files even if output already exists |
| `--max-files` | `10` | Maximum number of files to tag per run (`0` = unlimited) |

```bash
python3 scripts/run_question_tagger.py --force --max-files 0
```

### `scripts/run_golden_dataset.py` — Generate a golden dataset for retrieval testing

Scans all `.md` files in `data/silver/`, classifies each page by type (item,
character, location, etc.), and randomly selects up to 10 pages per type.
With 12 page types, the output contains at most 120 filenames. If a type has
fewer than 10 pages, all of them are included.

The selected filenames are written to `data/test-data/golden_pages.txt` (one
filename per line).

```bash
python3 scripts/run_golden_dataset.py
```

To get a reproducible selection, pass a seed:

```bash
python3 scripts/run_golden_dataset.py --seed 42
```

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--seed` | *(random)* | Random seed for reproducibility |
| `--per-type` | `10` | Maximum number of pages to select per type |
| `--input-dir` | `data/silver/` | Silver-tier input directory |
| `--output-dir` | `data/test-data/` | Output directory for the golden dataset file |

### `scripts/run_testset_generator.py` — Generate retrieval test sets

Reads the golden pages list from `data/test-data/golden_pages.txt`, picks one
random section per page from `data/silver-sections/`, and asks an LLM to
generate three types of question/answer pairs for retrieval evaluation:

- **Factoid** — asks for a specific name, place, or number.
- **Conceptual** — asks about an event, sequence, or how/why something happens.
- **Messy** — a casual, vague query like a quick Google search.

Results are written to three CSV files under `data/test-data/`:

| File | Contents |
|---|---|
| `factoid.csv` | Factoid Q/A pairs |
| `conceptual.csv` | Conceptual Q/A pairs |
| `messy.csv` | Messy Q/A pairs |

Each CSV has columns: `question`, `answer`, `section`, `page`.

Requires [Ollama](https://ollama.com/) running locally. Large prompts are
routed to Gemini (requires `GEMINI_API_KEY`).

```bash
python3 scripts/run_testset_generator.py
```

Configuration constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `GOLDEN_PAGES_PATH` | `data/test-data/golden_pages.txt` | Path to the golden pages file |
| `SECTIONS_DIR` | `data/silver-sections/` | Directory containing per-section Markdown files |
| `OUTPUT_DIR` | `data/test-data/` | Directory where CSV files are written |
| `MODEL_NAME` | `sandrock-model` | Local Ollama model to use |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite-preview` | Cloud model fallback for large prompts |

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--force` | off | Re-generate Q/A pairs even if already processed |
| `--max-questions` | `10` | Maximum number of pages to process per run (`0` = unlimited) |
| `--seed` | *(random)* | Random seed for reproducible section selection |

```bash
python3 scripts/run_testset_generator.py --force --max-questions 0 --seed 42
```

### `scripts/run_retrieval_eval.py` — Evaluate retrieval quality

Loads a test set CSV (factoid, conceptual, or messy), retrieves the top-k
results for each question from a ChromaDB collection, and computes evaluation
metrics:

- **Hit Rate@K** — fraction of queries where the expected document appears in
  the top-k results.
- **Page Hit Rate@K** — fraction of queries where *any* result from the expected
  page appears in the top-k. Useful for comparing section-level collections
  against page-level ones on equal footing.
- **NDCG@K** — mean Normalized Discounted Cumulative Gain. Rewards retrieving
  the expected document at higher ranks (score = 1/log2(rank+1)). A result at
  rank 1 scores 1.0, rank 2 scores ~0.63, rank 5 scores ~0.39.

For `summary` collections, a "hit" means the expected page filename appears in
the retrieved results. For `section-reverse-hyde` and
`section-tagged-reverse-hyde` collections, the expected section filename is
matched instead.

Requires Ollama running locally with `all-minilm` (embedding model).

```bash
python3 scripts/run_retrieval_eval.py --testset factoid --collection summary --k 5
```

Configuration constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `TESTSET_DIR` | `data/test-data/` | Directory containing test set CSV files |
| `DB_PATH` | `./sandrock_db` | Path to the ChromaDB database |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `all-minilm` | Ollama embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking |
| `RERANK_RETRIEVE_N` | `10` | Number of results to fetch before reranking |

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--testset` | *(prompted)* | Test set to evaluate: `factoid`, `conceptual`, or `messy` |
| `--collection` | *(prompted)* | Collection to test: `summary`, `section-reverse-hyde`, or `section-tagged-reverse-hyde` |
| `--k` | `5` | Number of top results to retrieve per query |
| `--metrics` | `hit_rate` | Space-separated list of metrics to compute |
| `--rerank` | `false` | Enable rerank mode: compare NDCG@K before/after cross-encoder reranking |
| `--ollama-url` | `http://localhost:11434` | Ollama server base URL |

```bash
python3 scripts/run_retrieval_eval.py --testset messy --collection section-reverse-hyde --k 10 --metrics hit_rate
```

**Rerank mode** fetches 10 results per query, computes metrics on the top-k
(before rerank), then reranks all 10 with a cross-encoder and computes metrics
on the reranked top-k. Automatically adds `ndcg` to the metrics list. Output
shows side-by-side before/after scores with deltas:

```bash
python3 scripts/run_retrieval_eval.py --testset factoid --collection summary --k 5 --rerank
```

Results are written to a CSV file in `data/test-data/` named
`eval_{testset}_{collection}_{metrics}_k{k}.csv` (or
`eval_{testset}_{collection}_rerank_{metrics}_k{k}.csv` in rerank mode). The
file contains:

- A **parameters section** (rows prefixed with `#`): testset, collection name,
  k, metrics used, and question count.
- **Per-query rows**: question, expected document, HIT or MISS, and the list of
  retrieved document IDs. In rerank mode, both before and after columns are
  included.
- **Summary scores**: one row per metric with the score at the given k.

### `scripts/diagnose_wiki.py` — Debug HTML-to-Markdown conversion

Inspects how a single raw HTML file is converted to Markdown, printing a verbose tag-by-tag trace. Saves the result to `diagnostic_result.md` in the current directory.

```bash
python3 scripts/diagnose_wiki.py path/to/file.html
```
