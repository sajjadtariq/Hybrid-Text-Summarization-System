# SummarizeX2

SummarizeX2 is a local text-summarization application with two methods:

- **Extractive (TextRank):** ranks source sentences and returns the strongest sentences unchanged.
- **Abstractive (DistilBART):** generates new wording with `sshleifer/distilbart-cnn-12-6` by default.

The React frontend calls a FastAPI backend. No hosted inference service or API key is required. Abstractive inference runs locally with PyTorch and downloads model weights from Hugging Face on first use.

## Features

- Extractive and abstractive summaries from the same interface
- Configurable target length from 20 to 250 words
- Input validation up to 50,000 characters
- Original word count, summary word count, compression ratio, and processing time
- Lazy transformer loading: extractive requests and health checks do not load DistilBART
- Hierarchical map-reduce summarization for inputs larger than the model context window
- Extractive fallback when abstractive output is too short or repetitive
- ROUGE-1, ROUGE-2, and ROUGE-L evaluation on eight bundled samples
- Docker Compose and native Windows workflows

## Architecture

```text
Browser
  |
  | HTTP/JSON
  v
React 19 + Vinext + Tailwind CSS
  |
  | POST /summarize
  v
FastAPI
  |-- request validation and response statistics
  |-- extractive -> tokenize -> normalize -> cosine graph -> PageRank
  `-- abstractive -> normalize -> lazy model load -> chunk -> map/reduce
                                                `-> TextRank fallback if degenerate

Evaluation CLI -> both summarizers -> rouge-score -> results.md + results.csv
```

### Extractive pipeline

`backend/summarizers/extractive.py` splits text into sentences, removes stopwords, applies lightweight rule-based normalization, represents sentences with term-frequency counters, and builds a weighted cosine-similarity graph. PageRank selects important sentences, which are returned in source order. The requested length is a soft upper bound because the method keeps sentences intact; multi-sentence input is capped at roughly 60% of the source length to avoid returning the full text.

### Abstractive pipeline

`backend/summarizers/abstractive.py` loads the configured Hugging Face tokenizer and sequence-to-sequence model on the first abstractive request. CUDA is used when PyTorch detects it; otherwise inference runs on CPU. Short inputs use one generation pass. Longer inputs are split into token chunks, summarized independently, and consolidated in a final pass. Repetitive or unusably short output falls back to the extractive summarizer.

## Requirements

For native Windows setup:

- CPython 3.11 or newer from python.org or the Python launcher (`py`)
- Node.js 22.13 or newer
- npm

For the container workflow:

- Docker Desktop with Docker Compose

The first abstractive request requires internet access to download model weights. It also needs substantially more memory and disk space than the extractive method.

## Windows setup (Command Prompt)

All commands below are for **Windows Command Prompt**, not PowerShell. Run them from a new `cmd.exe` window.

### 1. Backend

```bat
cd /d D:\SummarizeX2
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Keep that terminal open. Verify the API at:

- Health: `http://127.0.0.1:8000/health`
- OpenAPI documentation: `http://127.0.0.1:8000/docs`

If `.venv` already exists and works, do not recreate it; start with the install or Uvicorn command.

### 2. Frontend

Open a second Command Prompt:

```bat
cd /d D:\SummarizeX2\frontend
npm ci
npm run dev
```

Open `http://localhost:3000`.

The frontend already defaults to `http://localhost:8000`. To use a different backend URL:

```bat
cd /d D:\SummarizeX2\frontend
copy .env.example .env.local
notepad .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_URL` in `.env.local`, then restart the frontend.

## Docker Compose

From Windows Command Prompt:

```bat
cd /d D:\SummarizeX2
docker compose up --build
```

Open `http://localhost:3000`. The API is exposed at `http://localhost:8000`. Model files are stored in the named `model-cache` volume, so subsequent container starts reuse them.

After the model has been downloaded into that volume, offline mode can prevent Hugging Face network checks:

```bat
cd /d D:\SummarizeX2
set SUMMARIZER_OFFLINE=1
docker compose up
```

Offline mode fails if the selected model is not already cached.

## Configuration

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `SUMMARIZER_MODEL` | Backend | `sshleifer/distilbart-cnn-12-6` | Hugging Face model ID or local model path |
| `SUMMARIZER_OFFLINE` | Backend | `0` | Set to `1` to load only cached/local model files |
| `HF_HOME` | Hugging Face libraries | Library default; `/models/huggingface` in Docker | Model and tokenizer cache location |
| `NEXT_PUBLIC_API_URL` | Frontend | `http://localhost:8000` | Backend base URL used by the browser |

For a one-terminal native backend override in Command Prompt:

```bat
cd /d D:\SummarizeX2
set SUMMARIZER_MODEL=facebook/bart-large-cnn
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The larger BART model requires a separate download and more resources.

## API

### `GET /health`

Returns API status plus the configured abstractive model, offline setting, and whether the model is currently loaded. It does not trigger a model download.

### `POST /summarize`

Request fields:

| Field | Type | Constraints | Default |
|---|---|---|---|
| `text` | string | Nonblank, 1–50,000 characters | Required |
| `method` | string | `extractive` or `abstractive` | `extractive` |
| `length` | integer | 20–250 | `90` |

Example request:

```json
{
  "text": "Urban trees cool neighborhoods during heat waves. Trees also filter air pollution. Cities are expanding tree cover in areas with little shade.",
  "method": "extractive",
  "length": 40
}
```

Example response shape:

```json
{
  "summary": "Urban trees cool neighborhoods during heat waves. Cities are expanding tree cover in areas with little shade.",
  "method": "extractive",
  "stats": {
    "original_words": 22,
    "summary_words": 16,
    "compression_ratio": 0.714,
    "processing_ms": 1
  }
}
```

Model loading or inference failures return HTTP `503`. Invalid request data returns HTTP `422`.

## Evaluation

The evaluation set in `eval/sample_data.json` contains eight small, human-written news-style articles with reference summaries. `eval/run_evaluation.py` computes macro-averaged ROUGE F1 scores with stemming and writes:

- `eval/results.md`
- `eval/results.csv`

Install the development requirements first, then run from the project root:

```bat
cd /d D:\SummarizeX2
.venv\Scripts\python.exe -m eval.run_evaluation
```

Useful variants:

```bat
rem Fast baseline-only run
.venv\Scripts\python.exe -m eval.run_evaluation --method extractive

rem Abstractive-only run with the default 40-word target
.venv\Scripts\python.exe -m eval.run_evaluation --method abstractive

rem Both methods with a different target and output directory
.venv\Scripts\python.exe -m eval.run_evaluation --method both --length 60 --output eval
```

Available arguments:

- `--method extractive|abstractive|both` (default: `both`)
- `--length N` (default: `40` words)
- `--data PATH` (default: `eval/sample_data.json`)
- `--output PATH` (default: `eval`)

Each run overwrites `results.md` and `results.csv` in the selected output directory. A single-method run therefore writes only that method. The checked-in results currently contain the measured extractive baseline; the abstractive row remains unreported until the model evaluation completes:

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Samples |
|---|---:|---:|---:|---:|
| Extractive | 0.3684 | 0.1584 | 0.2996 | 8 |
| Abstractive | Not yet measured | Not yet measured | Not yet measured | 8 |

This is a small regression benchmark, not a claim of general model quality. ROUGE measures lexical overlap and does not establish factual consistency or human preference.

## Tests and frontend scripts

Run backend tests from the project root:

```bat
cd /d D:\SummarizeX2
.venv\Scripts\python.exe -m pytest backend\tests -q
```

Frontend commands run from `D:\SummarizeX2\frontend`:

```bat
npm run lint
npm run build
npm run start
```

`npm run start` serves the production build and should be run after `npm run build`. Use `npm run format` to format frontend source files.

Backend tests cover request validation, response statistics, TextRank preprocessing and compression, lazy model status, input normalization, hierarchical generation, and degenerate-output detection. The transformer tests use test doubles and do not download DistilBART weights.

## Project structure

```text
SummarizeX2/
|-- backend/
|   |-- main.py                         FastAPI application and schemas
|   |-- requirements.txt                Runtime Python dependencies
|   |-- requirements-dev.txt            Runtime plus test dependencies
|   |-- summarizers/
|   |   |-- extractive.py               TextRank implementation and CLI
|   |   `-- abstractive.py              Local transformer pipeline and CLI
|   `-- tests/                           Backend unit and API tests
|-- eval/
|   |-- sample_data.json                Eight-sample evaluation set
|   |-- run_evaluation.py               ROUGE evaluation CLI
|   |-- results.md                      Human-readable results
|   `-- results.csv                     Machine-readable results
|-- frontend/
|   |-- app/                             Vinext page, layout, and styles
|   |-- components/ui/                   Reusable controls
|   |-- public/                          Static assets
|   |-- .env.example                    Frontend API URL example
|   `-- package.json                    Frontend dependencies and scripts
|-- Dockerfile                           Backend and frontend build targets
`-- docker-compose.yml                  Local two-service stack and model cache
```

## Scope and limitations

- Text input only; there is no file upload or document parser.
- Submitted text is not persisted by the backend.
- Extractive length is a soft sentence-preserving budget.
- Abstractive length is converted from words to an approximate subword-token range.
- DistilBART is trained for news summarization and may perform poorly on other domains.
- Abstractive summaries can omit or introduce details; review important output against the source.
- The bundled evaluation is intentionally small and should be expanded before drawing comparative conclusions.
