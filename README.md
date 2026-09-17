# Hybrid Text Summarization System

**SummarizeX2** is a focused, local text-summarization application that compares two fundamentally different strategies: a transparent TextRank baseline that selects source sentences and a pretrained DistilBART transformer that writes a new summary. Text and inference stay on the user's machine; there are no LLM or hosted-inference API calls.

The dark, two-panel interface presents these methods as **Extract** and **Rewrite** so users can choose based on the desired result rather than the underlying implementation. The project is intentionally narrow enough to explain in an interview and complete enough to demonstrate NLP preprocessing, graph ranking, transformer inference, API design, evaluation, and frontend integration.

## Why compare both approaches?

**Extractive / TextRank** is fast, deterministic, and auditable. It represents sentences as normalized term-frequency vectors, connects similar sentences in a weighted graph, and applies PageRank. Its output is factually conservative because every sentence came from the input, but it can feel repetitive or disjointed.

**Abstractive / DistilBART** can compress and combine ideas into more natural prose. It is a smaller CNN/DailyMail-tuned BART variant, chosen as the default because it is more practical for CPU demos than `facebook/bart-large-cnn`. Long documents use hierarchical map-reduce summarization: every chunk first receives its own useful summary budget, then a final model pass consolidates those candidates and removes repetition. Its trade-offs are slower inference, greater memory use, and the possibility of unsupported details. Set `SUMMARIZER_MODEL=facebook/bart-large-cnn` to compare against the full model without changing code.

## Architecture

```text
┌──────────────────── React / Tailwind ────────────────────┐
│ text · method · target length  →  summary · local stats │
└──────────────────────────┬───────────────────────────────┘
                           │ POST /summarize
┌──────────────────────────▼───────────────────────────────┐
│                        FastAPI                           │
│  request validation · dispatch · compression · timing   │
└───────────────┬─────────────────────────┬────────────────┘
                │                         │
     ┌──────────▼──────────┐   ┌──────────▼───────────────┐
     │ TextRank            │   │ Local DistilBART         │
     │ tokenize → clean →  │   │ lazy load → tokenize →  │
     │ cosine graph → rank │   │ map chunks → consolidate│
     └──────────┬──────────┘   └──────────┬───────────────┘
                └──────────────┬──────────┘
                               ▼
                  ROUGE-1 / ROUGE-2 / ROUGE-L
```

## Quick start with Docker

Docker downloads the Python and Node dependencies; the first abstractive request also downloads model weights from Hugging Face Hub. Those weights persist in the `model-cache` volume, and every inference runs inside the local API container.

```bash
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000). API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

After the model has been cached, set `SUMMARIZER_OFFLINE=1` in the environment to prevent Hugging Face from checking the network:

```bash
SUMMARIZER_OFFLINE=1 docker compose up
```

## Run without Docker

Prerequisites: standard CPython 3.11+, Node.js 22+, and npm. PyTorch does not publish wheels for MSYS Python distributions.

### Windows Command Prompt

From the project root, start the API:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r backend\requirements-dev.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open a second Command Prompt, return to the project root, and start the frontend:

```bat
cd frontend
npm ci
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000).

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements-dev.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Copy `frontend/.env.example` to `frontend/.env.local` and change its value only if the API is not at `http://localhost:8000`.

## API

`POST /summarize` accepts raw text, a method, and a soft word budget:

```json
{
  "text": "Urban trees cool neighborhoods during heat waves. Trees also filter air pollution. Cities are expanding tree cover in areas with little shade.",
  "method": "extractive",
  "length": 40
}
```

Example response:

```json
{
  "summary": "Urban trees cool neighborhoods during heat waves. Cities are expanding tree cover in areas with little shade.",
  "method": "extractive",
  "stats": {
    "original_words": 21,
    "summary_words": 15,
    "compression_ratio": 0.714,
    "processing_ms": 1
  }
}
```

The word budget is intentionally soft for TextRank so it returns complete sentences. The abstractive path translates the requested word count to an approximate subword-token budget.

## Evaluation

The repository includes eight compact, human-written news-style samples so the benchmark is reviewable and quick to run. The evaluation script macro-averages ROUGE F1 values and writes both Markdown and CSV files.

```bash
python -m eval.run_evaluation
# faster baseline-only check
python -m eval.run_evaluation --method extractive
```

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Status |
|---|---:|---:|---:|---|
| TextRank | 0.3684 | 0.1584 | 0.2996 | Measured on the bundled 8-sample set |
| DistilBART | — | — | — | Pending the one-time model-weight download |

The extractive numbers were generated with `rouge-score` in the verified local runtime. The model host did not complete the large DistilBART weight transfer during the build session, so that row is deliberately blank rather than fabricated. Running the default command populates both rows in `eval/results.md` and `eval/results.csv`.

## Tests

```bash
pytest backend/tests -q
cd frontend && npm run build
```

The backend tests cover preprocessing, extractive behavior, request validation, response statistics, and the guarantee that blank abstractive input does not load the model.

## Project layout

```text
backend/
  main.py                    FastAPI application
  summarizers/
    extractive.py            from-scratch TextRank
    abstractive.py           lazy local transformer inference
  tests/
eval/
  sample_data.json           small reviewable benchmark
  run_evaluation.py          ROUGE runner and table writer
frontend/
  app/page.tsx               two-panel summarization interface
  app/globals.css            black-and-brown visual theme
Dockerfile                   shared multi-target image definition
docker-compose.yml           one-command local stack
```

## What I would improve with more time

1. **Evaluate on a recognized held-out corpus.** The bundled micro benchmark catches regressions, but a fixed CNN/DailyMail or XSum subset with recorded dataset and model revisions would make comparisons reproducible and more credible.
2. **Add factuality checks.** ROUGE rewards lexical overlap, not truthfulness. I would pair it with entity/number consistency checks and a semantic factuality metric, then review failure cases manually.
3. **Fine-tune for a target domain.** Rather than training a large model from scratch, I would use parameter-efficient fine-tuning on representative documents and evaluate whether gains justify the added maintenance and hardware cost.
4. **Make long-document chunking section-aware.** The current hierarchical pipeline prevents one oversized opening chunk from dominating, but a stronger version would split on headings and paragraphs, preserve section provenance, and allocate intermediate budgets according to salience.
5. **Calibrate length controls.** The UI asks for words while generation operates on subword tokens. I would learn a model-specific calibration curve and report the achieved budget distribution.
6. **Add multi-document provenance.** For research or briefing workflows, I would preserve sentence-level citations and surface disagreements across sources rather than producing one unqualified narrative.
7. **Measure operational behavior.** Startup time, peak memory, tokens per second, and CPU/GPU comparisons matter for a local product and would turn model selection into an evidence-based engineering decision.

These improvements are intentionally omitted from the MVP: each adds a meaningful research or product question, and none is required to demonstrate the end-to-end fundamentals.
