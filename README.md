# Playlist → Book Pipeline

Convert YouTube playlist transcripts into a professionally formatted technical book — fully automated with Google Gemini.

```
YouTube Transcripts → Clean & Chunk → LLM Writes Chapters → Markdown + PDF Book
```

## Features

- **End-to-end pipeline** — pre-fetched transcripts → structured chapters → polished PDF
- **Gemini-powered** — uses `gemini-2.5-flash` for fast, high-quality generation
- **Token-aware chunking** — automatically splits large transcripts while preserving sentence boundaries
- **Caching & resumability** — chapters are cached to disk; interrupted runs pick up where they left off
- **Parallel processing** — writes multiple chapters concurrently (configurable batch size)
- **Book-quality PDF** — professional typography, code highlighting, title page, and table of contents

## Quick Start

### Prerequisites

- Python 3.10+
- A [Gemini Paid API key](https://aistudio.google.com/apikey)

### Setup

```bash
git clone <repo-url> && cd YouTube-playlist-to-PDF
chmod +x setup.sh && ./setup.sh
```

This creates a virtual environment, installs dependencies, and generates a `.env` file. Add your API key:

```bash
# Edit .env and set your key
GEMINI_API_KEY=your_key_here
```

### Run

```bash
source .venv/bin/activate

# Preview the generated table of contents (no LLM chapter writing)
python main.py --dry-run

# Full run — generates Markdown + PDF
python main.py
```

### Docker

No local Python setup needed — just Docker:

```bash
chmod +x run.sh
./run.sh              # Full pipeline
./run.sh --dry-run    # Preview TOC only
```

The script builds the image, mounts `output/` for results, and reads your `.env` for API keys.

## Usage

```bash
python main.py [OPTIONS]
```

| Flag | Description |
|---|---|
| `--dry-run` | Load data and generate TOC only (skip chapter writing) |
| `--config PATH` | Use a custom YAML config file |
| `-v`, `--verbose` | Enable debug logging |

## How It Works

1. **Load metadata** — reads video titles, durations, and chapter markers from `data/playlist_metadata/`
2. **Generate TOC** — Gemini proposes a book structure by grouping related videos into chapters
3. **Write chapters** — for each chapter, loads transcripts from `data/transcripts/`, cleans filler words, splits into token-aware chunks, and sends each chunk to Gemini with rolling context for narrative continuity
4. **Export** — assembles all chapters into a Markdown manuscript and renders a styled PDF via WeasyPrint

## Configuration

All settings live in `config/default.yaml`:

| Setting | Default | Description |
|---|---|---|
| `llm.gemini.model` | `gemini-2.5-flash` | Gemini model to use |
| `llm.gemini.temperature` | `0.3` | Generation temperature (lower = more consistent) |
| `chunking.target_chunk_tokens` | `12000` | Max tokens per chunk sent to the LLM |
| `chunking.overlap_sentences` | `3` | Sentence overlap between consecutive chunks |
| `processing.parallel_chapters` | `5` | Chapters to write concurrently per batch |
| `processing.filler_words` | `[um, uh, ...]` | Filler words removed during cleaning |

## Output

| Path | Description |
|---|---|
| `output/manuscript.md` | Complete book in Markdown |
| `output/manuscript.pdf` | Formatted PDF with title page and TOC |
| `output/chapters/` | Individual cached chapter files |

## Project Structure

```
├── main.py                        # CLI entry point & pipeline orchestrator
├── setup.sh                       # One-command setup script
├── run.sh                         # Docker runner
├── Dockerfile
├── config/
│   └── default.yaml               # Pipeline settings
├── data/
│   ├── transcripts/               # Pre-fetched transcript JSON files
│   └── playlist_metadata/         # Video metadata (titles, chapters, durations)
├── src/
│   ├── config.py                  # YAML config loader with dataclasses
│   ├── transcript/
│   │   └── fetcher.py             # Loads transcripts & metadata from data/
│   ├── processing/
│   │   ├── cleaner.py             # Filler removal, whitespace normalization
│   │   └── chunker.py             # Token-aware text splitting (tiktoken)
│   ├── llm/
│   │   ├── base.py                # Abstract LLM provider with retry logic
│   │   ├── gemini_provider.py     # Google Gemini integration
│   │   └── factory.py             # Provider factory
│   └── book/
│       ├── structurer.py          # TOC generation & chapter writing
│       └── exporter.py            # Markdown & PDF export (WeasyPrint)
├── tests/
│   ├── test_cleaner.py
│   └── test_chunker.py
├── requirements.txt
└── .env.example
```

## Testing

```bash
pytest tests/
```

## License

MIT
