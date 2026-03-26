# The Commerce Guild Assistant

A RAG-powered knowledge assistant for the My Time series (Sandrock & Portia) using multi-representation indexing.

## Project Setup

- **Python**: >= 3.12 (use `python3` to run scripts)
- **Package manager**: Poetry (build backend: `poetry.core.masonry.api`)
- **Activate environment**: `eval "$(poetry env activate)"`
- **Add dependencies**: `poetry add <package>`
- **Run tests**: `pytest` (dev dependencies include pytest and pytest-mock)

## Directory Structure

```
the-commerce-guild-assistant/
  src/guild_assistant/          # Main package source code
    scraper/                    # Wiki scraper module
      discoverer.py             # Thread that discovers wiki page IDs via MediaWiki API
      worker.py                 # Thread that fetches pages, cleans HTML, converts to Markdown
    utils/                      # Shared utilities (model adapters, routing, classification)
      model_adapter.py          # ABC base class, OllamaAdapter, and GeminiAdapter for LLM access (Adapter pattern)
      router.py                 # ModelRouter: token-based dispatch to local or cloud model via tiktoken
      page_classifier.py        # PageType enum, classify_page(), classify_file(), extract_heading_titles()
    rag_setup/                  # RAG setup module (summarization + embedding pipeline)
      summarizer.py             # Reads silver .md files, generates LLM summaries, writes to data/summaries
      section_breaker.py        # Breaks silver .md files into type-aware chunks in data/silver-sections (uses page_classifier for type detection)
      question_generator.py     # Reads silver-section .md files, generates hypothetical questions (Reverse HyDE), writes to data/reverse-hyde
      question_tagger.py        # Prepends metadata tags (page slug + L2/L3 headings) to reverse-hyde questions, writes to data/tagged-reverse-hyde
      embedder.py               # Reads .md files from a sources directory, embeds with all-MiniLM-L6-v2, persists to ChromaDB
    rag_test/                   # RAG test data generation (golden datasets + test sets)
      golden_dataset.py         # Stratified sampling of silver-tier pages by type for retrieval testing
      testset_generator.py      # Generates factoid/conceptual/messy Q/A pairs from golden pages for retrieval evaluation
    rag/                        # RAG query pipeline (retrieval + augmentation + generation)
      retriever.py              # Retriever: embeds a query, finds top-k matches from ChromaDB
      reranker.py               # Reranker: cross-encoder reranking of retrieval results
      context_builder.py        # ContextBuilder: reads full silver-tier pages, formats augmented LLM prompt
      pipeline.py               # QueryPipeline: orchestrates retrieve → rerank → augment → generate with pluggable transforms
  src/guild_assistant_web/      # Web UI package (Chainlit frontend, separate from backend)
    app.py                      # Chainlit app: chat start, settings update, message handlers
    pipeline_factory.py         # Factory that builds a QueryPipeline for a given retrieval mode
    chainlit.md                 # Chainlit welcome/landing page content
  scripts/                      # Runnable entry-point scripts
    run_scraper.py              # Main scraper entry point (spawns discoverer + worker threads)
    diagnose_wiki.py            # Diagnostic tool for debugging HTML-to-Markdown conversion
    run_summarizer.py           # Generates LLM summaries for silver-tier Markdown files via Ollama
    preview_model_routing.py    # Dry-run: prints which model (local/Gemini) would be used per file
    run_section_breaker.py      # Breaks silver .md files into per-section files in data/silver-sections
    run_question_tagger.py      # Tags reverse-hyde questions with page slug + L2/L3 headings, writes to data/tagged-reverse-hyde
    run_embedder.py             # Embeds summaries into ChromaDB (sandrock_db/); run after run_summarizer.py
    run_chat.py                 # Interactive REPL chatbot powered by the RAG query pipeline
    run_web.py                  # Launches the Chainlit web UI for the RAG chatbot
    run_golden_dataset.py       # Generates a golden dataset (stratified sample) for retrieval testing
    run_testset_generator.py    # Generates factoid/conceptual/messy Q/A test sets from golden pages
    run_retrieval_eval.py       # Evaluates retrieval quality (Hit Rate@K) against labelled test sets
  tests/                        # Unit tests (pytest)
    benchmark/                  # Retrieval evaluation framework
      testset_loader.py         # TestCase dataclass and load_testset() for reading test CSV files
      metrics.py                # Metric functions (hit_rate) and METRIC_REGISTRY for retrieval evaluation
      retrieval_evaluator.py    # RetrievalEvaluator: runs test cases through a Retriever and computes metrics
    test_discoverer.py          # Tests for the Discoverer thread
    test_worker.py              # Tests for the Worker thread
    test_model_adapter.py       # Tests for ModelAdapter ABC, OllamaAdapter, GeminiAdapter, RerankerAdapter, and CrossEncoderRerankerAdapter
    test_summarizer.py          # Tests for the Summarizer class
    test_section_breaker.py     # Tests for the SectionBreaker class (page classification, chunking, overview grouping)
    test_question_generator.py  # Tests for the QuestionGenerator class
    test_question_tagger.py     # Tests for the QuestionTagger class (tagging, heading extraction, skip/force)
    test_router.py              # Tests for ModelRouter token-based routing
    test_embedder.py            # Tests for the Embedder class
    test_retriever.py           # Tests for the Retriever class
    test_context_builder.py     # Tests for the ContextBuilder class
    test_reranker.py            # Tests for the Reranker class
    test_pipeline.py            # Tests for the QueryPipeline class (including reranker integration)
    test_page_classifier.py     # Tests for PageType, classify_page, classify_file, extract_heading_titles
    test_golden_dataset.py      # Tests for the golden dataset generator (stratified sampling, file output)
    test_testset_generator.py   # Tests for the TestsetGenerator class (Q/A generation, CSV output, skip/force logic)
    test_testset_loader.py      # Tests for the TestCase dataclass and load_testset()
    test_metrics.py             # Tests for hit_rate, page_hit_rate, ndcg, _page_slug, and METRIC_REGISTRY
    test_retrieval_evaluator.py # Tests for the RetrievalEvaluator class (summary/section matching, edge cases, rerank mode)
    test_pipeline_factory.py    # Tests for the pipeline factory (mode validation, collection mapping, custom params)
    test_chainlit_app.py        # Tests for the Chainlit app handlers (session, settings, message)
  data/                         # Scraped output (git-ignored)
    raw/                        # Raw HTML and cleaned Markdown files from the scraper
    silver/                     # Filtered full-text Markdown files (source of truth for retrieval)
    silver-sections/            # Per-section splits of silver files (input for question generation)
    summaries/                  # LLM-generated summaries of silver-tier files (embedded into ChromaDB)
    reverse-hyde/               # Hypothetical questions per silver-section file (Reverse HyDE indexing)
    tagged-reverse-hyde/        # Reverse-hyde questions prepended with metadata tags (page slug + L2/L3 headings)
    test-data/                  # Golden dataset and test artifacts for retrieval evaluation
  sandrock_db/                  # Persisted ChromaDB vector store (git-ignored)
  pyproject.toml                # Project metadata and dependencies
  poetry.lock                   # Locked dependency versions
```

## Key Components

### Scraper Pipeline (`src/guild_assistant/scraper/`)

The scraper uses a producer-consumer threading pattern:

- **`discoverer.py`** - `Discoverer` thread queries the MediaWiki `allpages` API endpoint with pagination to enumerate all wiki pages. Pushes page IDs into a shared `Queue`. Sends `None` sentinels (one per worker) when done.
- **`worker.py`** - `Worker` threads pull page IDs from the queue, fetch rendered HTML via the MediaWiki `parse` API, clean it (strip galleries, navboxes, TOC, edit links, fix tables, resolve images), and convert to Markdown using `markdownify`. Outputs both `.html` and `.md` files to `data/raw/`.

### Shared Utilities (`src/guild_assistant/utils/`)

- **`model_adapter.py`** - `ModelAdapter` abstract base class defining the `generate(prompt) -> str` interface. `OllamaAdapter` calls a local Ollama instance's `/api/generate` endpoint. `GeminiAdapter` calls the Google Generative Language REST API using `GEMINI_API_KEY` from the environment. `EmbeddingAdapter` ABC with `OllamaEmbeddingAdapter` for embedding via Ollama `/api/embed`. `RerankerAdapter` ABC defining the `score(pairs) -> list[float]` interface with `CrossEncoderRerankerAdapter` using a `sentence-transformers` `CrossEncoder` model. New backends can be added by subclassing `ModelAdapter`, `EmbeddingAdapter`, or `RerankerAdapter`.
- **`router.py`** - `ModelRouter` class (also a `ModelAdapter`) that routes prompts based on token count. Uses `tiktoken` (`cl100k_base` encoding) to count tokens; prompts with fewer than the configured threshold go to the local model, larger prompts go to the cloud (Gemini) model. Default threshold is 11 000 tokens. The `route(prompt)` method returns the selected adapter without calling it.
- **`page_classifier.py`** - `PageType` enum and `classify_page(filename, heading_titles)` function for determining wiki page types (item, character, location, monster, mission, dialogue, buyback, event, store, region, festival, generic). Also provides `extract_heading_titles(content)` to pull ATX/setext headings from Markdown and `classify_file(file_path)` as a convenience wrapper. Used by both `section_breaker.py` and `golden_dataset.py`.

### RAG Setup (`src/guild_assistant/rag_setup/`)

- **`summarizer.py`** - `Summarizer` class that reads `.md` files from an input directory, builds a structured prompt, sends it to a `ModelRouter` (which dispatches to local or cloud based on token count), and writes the summary to an output directory. Supports skip-if-exists and force-regeneration modes.
- **`section_breaker.py`** - `SectionBreaker` class that reads `.md` files from an input directory (typically `data/silver/`), classifies each page by type using `classify_page()` from `utils.page_classifier` (item, character, location, monster, mission, dialogue, buyback, event, store, region, festival, or generic), and produces chunks following type-specific rules. Each L2 section (with all L3+ subsections) forms one chunk. Sections designated as "Overview" for the page's type are merged into a single Overview chunk; remaining L2 sections become standalone chunks. Output filenames follow `<page_slug>-<chunk_name>.md` where `chunk_name` is `overview` or the slugified L2 heading. Chunks containing only headings (no body text) are skipped. `break_all(force, max_files)` supports skip-if-exists, force re-breaking, and a per-run cap (`max_files=10` by default; `0` = unlimited).
- **`question_generator.py`** - `QuestionGenerator` class that reads per-section `.md` files from an input directory (typically `data/silver-sections/`), parses the page and section titles from the filename, and sends context-aware prompts to a `ModelRouter`. For overview chunks (`*-overview.md`), generates 5 questions using only the overview content. For all other chunks, generates 1-3 questions and includes the page's overview chunk as read-only context so the LLM understands the target section better (the LLM is instructed to generate questions only about the target section, not the overview). Falls back to a simpler prompt if no overview file exists. The output contains questions only — no section headers, numbering, or filler. Supports skip-if-exists and force-regeneration modes via `generate_all(force, max_files)`. Exports: `_is_overview()`, `_overview_path_for()`, `_parse_section_filename()`.
- **`question_tagger.py`** - `QuestionTagger` class that reads reverse-hyde question files from `data/reverse-hyde/` and the corresponding silver-section files from `data/silver-sections/`. For each file, extracts the page slug from the filename and all L2/L3 heading titles from the section content, then prepends a bracketed tag line (e.g. `[yakmel | Obtaining | From NPCs]`) to the questions. Writes tagged copies to `data/tagged-reverse-hyde/`. No LLM calls — pure text processing. Supports skip-if-exists and force modes via `tag_all(force, max_files)`. Exports: `_extract_l2_l3_titles()`, `_build_tag_line()`, `_parse_page_slug()`.
- **`embedder.py`** - `Embedder` class that reads `.md` files from a `sources_dir`, calls an `EmbeddingAdapter` to compute vectors, and upserts into a named ChromaDB `PersistentClient` collection (caller-supplied via `collection_name`). Accepts an `EmbeddingAdapter` dependency — no model logic lives here. Each entry stores `filename`, `summary_path`, and `silver_path` metadata. The `silver_path` is the key for downstream retrieval: a similarity match on the source embedding resolves to the full-text file in `data/silver/`. `embed_all(force, max_records)` supports skip-if-exists, force re-embedding, and a per-run record cap (`max_records=10` by default; `0` = unlimited).

### RAG Query Pipeline (`src/guild_assistant/rag/`)

- **`retriever.py`** - `Retriever` class that embeds a user query via an `EmbeddingAdapter` and queries a ChromaDB collection for the top-k nearest matches. Returns a list of `RetrievalResult` dataclasses containing `doc_id`, `summary`, `silver_path`, and `distance`.
- **`reranker.py`** - `Reranker` class that scores retrieval results against the user query using a `RerankerAdapter`. For each result, reads the actual silver-tier document content and builds `(query, document)` pairs for the adapter. Returns the top `top_n` results sorted by score. Skips results with missing silver files. Traced via LangSmith (`@traceable`).
- **`context_builder.py`** - `ContextBuilder` class that reads full-text silver-tier files referenced by retrieval results and formats them into a structured prompt. The prompt instructs the LLM to answer only from the provided documents and to say "I don't have enough information" when the documents are insufficient.
- **`pipeline.py`** - `QueryPipeline` class that orchestrates the full RAG flow: query → retrieve → rerank (optional) → augment → generate. Accepts any `ModelAdapter` for generation (typically a `ModelRouter`). Accepts an optional `Reranker` for cross-encoder reranking between retrieval and context building. The reranker receives the original user query (not the transformed query). Supports pluggable query transforms via `add_query_transform(fn)` for pre-processing (e.g. query expansion, spelling correction).

### Web UI (`src/guild_assistant_web/`)

- **`pipeline_factory.py`** - `create_pipeline(mode, *, db_path, ollama_url, gemini_model, embedding_model, reranker_model)` factory function that constructs a fully-configured `QueryPipeline` for a given retrieval mode (`"summary"`, `"section-reverse-hyde"`, or `"section-tagged-reverse-hyde"`). Extracts the pipeline-construction logic from `run_chat.py` into a reusable function shared by both the CLI and web UI. Raises `ValueError` for invalid modes. Exports: `VALID_MODES`, `_MODE_CONFIG`, `create_pipeline()`.
- **`app.py`** - Chainlit application with three handlers: `on_chat_start()` initialises the session with a default `"summary"` pipeline and renders a `ChatSettings` dropdown for mode selection; `on_settings_update()` rebuilds the pipeline when the user switches mode; `on_message()` runs the user's question through the pipeline via `cl.make_async` (thread pool) and sends the answer. Pipelines are stored per-session via `cl.user_session`.
- **`chainlit.md`** - Welcome page content rendered by Chainlit when no messages exist.

### RAG Test Data Generation (`src/guild_assistant/rag_test/`)

- **`golden_dataset.py`** - `select_golden_pages(silver_dir, per_type, seed)` classifies every `.md` file in a silver directory by page type (via `utils.page_classifier`) and randomly samples up to `per_type` (default 10) pages per type. With 12 types, this yields at most 120 pages. Types with fewer pages than the limit are included in full. `write_golden_dataset(silver_dir, output_dir, output_filename, per_type, seed)` wraps selection and writes the sorted filenames to a text file (one per line). Both functions accept an optional `seed` for reproducibility.
- **`testset_generator.py`** - `TestsetGenerator` class that reads a golden-pages file, finds per-page sections in `data/silver-sections/`, randomly picks one section per page, and asks an LLM (via `ModelRouter`) to generate three types of Q/A pairs: factoid (specific name/place/number), conceptual (event/sequence/how/why), and messy (casual vague query). Parses the structured LLM response and writes results to three CSV files (`factoid.csv`, `conceptual.csv`, `messy.csv`) under the output directory. Each CSV row contains `question`, `answer`, `section`, `page`. `generate_all(force, max_questions)` supports skip-if-exists (checks `factoid.csv` for already-processed page slugs), force re-generation, and a per-run cap (`max_questions=10` by default; `0` = unlimited). Accepts an optional `seed` for reproducible section selection. Exports: `_parse_response()`, `_find_sections()`, `_page_slug_from_filename()`, `_parse_section_filename()`, `_load_processed_pages()`.

### Benchmark (`tests/benchmark/`)

- **`testset_loader.py`** - `TestCase` dataclass (`question`, `answer`, `section`, `page`) and `load_testset(csv_path)` function for reading test set CSV files into a list of `TestCase` instances. Shared by the retrieval evaluator and any future evaluation tools.
- **`metrics.py`** - Retrieval evaluation metrics. `QueryResult` dataclass holds `question`, `expected_id`, and `retrieved_ids`. `hit_rate(results)` computes the fraction of queries where the expected document appears in the retrieved set. `page_hit_rate(results)` computes the fraction of queries where any result from the expected *page* appears, regardless of section — useful for comparing section-level collections against page-level ones on equal footing. `ndcg(results)` computes mean Normalized Discounted Cumulative Gain — rewards retrieving the expected document at higher ranks (1/log2(rank+1)). `METRIC_REGISTRY` maps metric names to functions; new metrics can be added by defining a function and registering it.
- **`retrieval_evaluator.py`** - `RetrievalEvaluator` class that runs test cases through a `Retriever` and computes metrics. Automatically matches on `page` for summary collections or `section` for section-reverse-hyde and section-tagged-reverse-hyde collections by extracting the basename from `silver_path`. `run(test_cases, metrics)` returns a dict of `{metric_name: score}`. Optionally accepts a `Reranker` and `retrieve_n` for rerank evaluation: `run_with_rerank(test_cases, metrics)` fetches `retrieve_n` results (default 10), computes metrics on the top-k before reranking, then reranks and computes metrics on the top-k after — returning `(before_scores, after_scores, before_results, after_results)` for comparison.

### Scripts (`scripts/`)

- **`run_scraper.py`** - Launches the full scraping pipeline. Configurable constants at the top: `OUTPUT_DIR`, `MAX_PAGES` (0 = unlimited), `NUM_WORKERS` (default 5). Run with: `python3 scripts/run_scraper.py`
- **`diagnose_wiki.py`** - Debug tool for inspecting how a single HTML file converts to Markdown. Run with: `python3 scripts/diagnose_wiki.py path/to/file.html`
- **`copy_to_silver.py`** - Copies filtered Markdown files from `data/raw/` to `data/silver/`, excluding meta/changelog pages. For buyback files, prepends a `# <filename>` title (without `.md` suffix). Run with: `python3 scripts/copy_to_silver.py`
- **`run_summarizer.py`** - Generates LLM summaries for all silver-tier Markdown files using `ModelRouter` to dispatch each prompt to the appropriate model. Configurable constants: `INPUT_DIR`, `OUTPUT_DIR`, `MODEL_NAME` (local Ollama model name), `OLLAMA_URL`, `GEMINI_MODEL` (cloud fallback). Requires Ollama running locally and `GEMINI_API_KEY` in the environment (or `.env`) for large-file routing. Run with: `python3 scripts/run_summarizer.py`. Use `--force` to regenerate all summaries even if they already exist.
- **`preview_model_routing.py`** - Dry-run companion to `run_summarizer.py`. Reads every silver-tier `.md` file, counts tokens using the same `ModelRouter` logic, and prints a table showing which model (local llama or Gemini) would be used for each file. No model API calls are made. Run with: `python3 scripts/preview_model_routing.py`
- **`run_question_tagger.py`** - Prepends metadata tags to reverse-hyde question files. Reads question files from `data/reverse-hyde/` and corresponding section files from `data/silver-sections/`, extracts the page slug and all L2/L3 heading titles, and writes tagged copies to `data/tagged-reverse-hyde/`. No LLM calls — pure text processing. Tag format: `[page_slug | Heading1 | Heading2 | ...]`. CLI flags: `--force` (re-tag existing), `--max-files` (cap per run, default `10`; `0` = unlimited). Run with: `python3 scripts/run_question_tagger.py`.
- **`run_embedder.py`** - Constructs an `OllamaEmbeddingAdapter` (`all-minilm`) and an `Embedder`, then embeds content into `./sandrock_db`. Requires `--mode` to select the embedding strategy: `summary` reads from `data/summaries/` and writes to the `sandrock_wiki_summary` collection (silver links to `data/silver/`); `section-reverse-hyde` reads from `data/reverse-hyde/` and writes to the `sandrock_wiki_section_reverse_hyde` collection (silver links to `data/silver-sections/`); `section-tagged-reverse-hyde` reads from `data/tagged-reverse-hyde/` and writes to the `sandrock_wiki_section_tagged_reverse_hyde` collection (silver links to `data/silver-sections/`). If `--mode` is omitted, the script prompts interactively. Configurable constants: `DB_PATH`, `OLLAMA_URL`, `EMBEDDING_MODEL`. CLI flags: `--mode` (required; `summary`, `section-reverse-hyde`, or `section-tagged-reverse-hyde`), `--force` (re-embed existing), `--max-records` (cap per run, default `10`; `0` = unlimited), `--ollama-url`. Run with: `python3 scripts/run_embedder.py --mode summary`.
- **`drop_collection.py`** - Deletes a named ChromaDB collection and all its embeddings from the local vector store. Defaults to `sandrock_wiki_summary`. CLI flags: `--collection` (collection to drop), `--db-path` (database directory). Exits with an error if the collection does not exist. Run with: `python3 scripts/drop_collection.py`.
- **`run_section_breaker.py`** - Breaks silver-tier Markdown files into type-aware chunks. Reads from `data/silver/`, writes to `data/silver-sections/`. Each page is classified by type (item, character, location, etc.) and chunked accordingly: type-designated overview sections are merged into an Overview chunk (`<page_slug>-overview.md`), remaining L2 sections become standalone chunks (`<page_slug>-<section_slug>.md`). Chunks with only headings are skipped. CLI flags: `--force` (re-break existing), `--max-files` (cap per run, default `10`; `0` = unlimited). Run with: `python3 scripts/run_section_breaker.py`.
- **`run_question_generator.py`** - Generates hypothetical questions for all silver-section Markdown files using `ModelRouter` to dispatch each prompt. The prompt includes page and section titles (derived from the filename) for LLM context and asks for 1-3 questions per section. Output contains questions only (one per line), stored under `data/reverse-hyde/` with the same filename as the source section. Configurable constants: `INPUT_DIR` (`data/silver-sections/`), `OUTPUT_DIR`, `MODEL_NAME`, `OLLAMA_URL`, `GEMINI_MODEL`. CLI flags: `--force` (re-generate existing), `--max-files` (cap per run, default `10`; `0` = unlimited). Run with: `python3 scripts/run_question_generator.py`.
- **`run_chat.py`** - Interactive REPL chatbot powered by the RAG query pipeline. Embeds user questions, retrieves top-5 candidates from ChromaDB, reranks them with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to select the best 3, augments a prompt with the reranked content, and generates answers via `GeminiAdapter`. Requires `--mode` to select which collection to query: `summary` retrieves from `sandrock_wiki_summary` (full pages from `data/silver/`); `section-reverse-hyde` retrieves from `sandrock_wiki_section_reverse_hyde` (per-section chunks from `data/silver-sections/`); `section-tagged-reverse-hyde` retrieves from `sandrock_wiki_section_tagged_reverse_hyde` (tagged per-section chunks from `data/silver-sections/`). Prompts interactively if `--mode` is omitted. CLI flags: `--mode` (required; `summary`, `section-reverse-hyde`, or `section-tagged-reverse-hyde`), `--ollama-url`. Configurable constants: `DB_PATH`, `OLLAMA_URL`, `GEMINI_MODEL`, `EMBEDDING_MODEL`, `RERANKER_MODEL`. Requires Ollama running locally. Run with: `python3 scripts/run_chat.py --mode summary`. Type `quit` or `exit` to stop.
- **`run_web.py`** - Launches the Chainlit web UI for the RAG chatbot. Starts a local web server with an interactive chat interface. Users can switch between `summary`, `section-reverse-hyde`, and `section-tagged-reverse-hyde` retrieval modes via the settings panel (gear icon). Uses `pipeline_factory.create_pipeline()` from `guild_assistant_web`. CLI flags: `--host` (default `localhost`), `--port` (default `8000`), `-w`/`--watch` (auto-reload). Requires Ollama running locally with `all-minilm` and `GEMINI_API_KEY` in the environment. Run with: `python3 scripts/run_web.py`. Or directly: `chainlit run src/guild_assistant_web/app.py -w`.
- **`run_golden_dataset.py`** - Generates a golden dataset for retrieval testing. Scans `data/silver/`, classifies each page by type, and randomly selects up to 10 pages per type (12 types × 10 = 120 max). Writes selected filenames to `data/test-data/golden_pages.txt`. CLI flags: `--seed` (RNG seed for reproducibility), `--per-type` (max pages per type, default 10), `--input-dir` (default `data/silver/`), `--output-dir` (default `data/test-data/`). Run with: `python3 scripts/run_golden_dataset.py`. Use `--seed 42` for reproducible output.
- **`run_testset_generator.py`** - Generates retrieval test sets from golden pages. Reads `data/test-data/golden_pages.txt`, picks one random section per page from `data/silver-sections/`, and asks an LLM to generate factoid, conceptual, and messy Q/A pairs. Results are written to three CSV files (`factoid.csv`, `conceptual.csv`, `messy.csv`) under `data/test-data/`. Uses `ModelRouter` to dispatch prompts to local Ollama (`sandrock-model`) or cloud Gemini (`gemini-3.1-flash-lite-preview`) based on token count. CLI flags: `--force` (re-generate existing), `--max-questions` (cap per run, default `10`; `0` = unlimited), `--seed` (RNG seed for section selection). Run with: `python3 scripts/run_testset_generator.py`.
- **`run_retrieval_eval.py`** - Evaluates retrieval quality against a labelled test set. Loads a test CSV (factoid, conceptual, or messy), retrieves top-k results for each question from a ChromaDB collection, and computes metrics (e.g. Hit Rate@K). For summary collections, matches on page filename; for section-reverse-hyde and section-tagged-reverse-hyde collections, matches on section filename. Prompts interactively if `--testset` or `--collection` is omitted. Writes per-query results to a CSV file in `data/test-data/` named `eval_{testset}_{collection}_{metrics}_k{k}.csv`, containing a parameters section (testset, collection, k, metrics, question count), per-query rows (question, expected doc, HIT/MISS, retrieved docs), and summary scores. Supports a `--rerank` flag for cross-encoder rerank evaluation: fetches 10 results, reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2`, and reports NDCG@K before and after reranking for comparison (automatically adds `ndcg` to the metrics list). Configurable constants: `TESTSET_DIR`, `DB_PATH`, `OLLAMA_URL`, `EMBEDDING_MODEL`, `RERANKER_MODEL`, `RERANK_RETRIEVE_N`. CLI flags: `--testset` (prompted if omitted; `factoid`, `conceptual`, or `messy`), `--collection` (prompted if omitted; `summary`, `section-reverse-hyde`, or `section-tagged-reverse-hyde`), `--k` (top-k, default `5`), `--metrics` (space-separated list, default `hit_rate`), `--rerank` (enable rerank comparison mode), `--ollama-url`. Run with: `python3 scripts/run_retrieval_eval.py --testset factoid --collection summary --k 5`. Rerank mode: `python3 scripts/run_retrieval_eval.py --testset factoid --collection summary --k 5 --rerank`.

## Key Dependencies

- `requests` - HTTP client for MediaWiki API calls
- `beautifulsoup4` / `bs4` - HTML parsing and DOM manipulation
- `markdownify` - HTML-to-Markdown conversion
- `python-dotenv` - Environment variable loading (`.env` → `GEMINI_API_KEY`)
- `tiktoken` - Token counting for prompt routing (cl100k_base encoding)
- `chromadb` - Local vector store for persisting embeddings (`PersistentClient`, no server required)
- `sentence-transformers` - Cross-encoder models for reranking retrieval results (`CrossEncoder` class)
- `requests` (also used by scraper/summarizer) - `OllamaEmbeddingAdapter` calls the local Ollama `/api/embed` endpoint directly via `requests`; requires Ollama running with `all-minilm` pulled
- `chainlit` - Web UI framework for building conversational interfaces; powers the browser-based chat in `guild_assistant_web`

## Documentation

- Whenever a new script is added to `scripts/`, its description **must** be added to both `README.md` (as a full `###` section with usage examples and a CLI flags table) and `CLAUDE.md` (as a bullet under **Scripts (`scripts/`)**).

## Conventions

- The wiki target is `https://mytimeatsandrock.fandom.com` (hardcoded in scraper modules as `_BASE_URL`).
- File naming uses snake_case derived from wiki page titles (see `Worker._title_to_snake`).
- Redirect pages are detected and skipped during scraping.
- The `data/` directory is git-ignored; scraped content is not committed.
- Logging uses Python's `logging` module with thread-name-aware formatting.
- The project uses threading (not asyncio) for concurrent scraping.
