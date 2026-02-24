# Playlist → Book Pipeline

Convert pre-fetched YouTube playlist transcripts into a professionally formatted technical book manuscript, fully automated.

## Features

- **End-to-end pipeline**: pre-fetched transcripts → chapters → polished PDF
- **Works out of the box**: uses Gemini by default
- **Large content handling**: automatic transcript chunking with token-aware splitting
- **Caching & resumability**: chapters are cached; a failed run can resume
- **Optional verification**: cross-check chapters against source transcripts using the Gemini LLM
- **Book-quality output**: professional PDF with proper typography, code highlighting, and page layout

## Quick Start (One Command)

```bash
git clone <repo-url> && cd playlist_to_book
chmod +x setup.sh && ./setup.sh
```

This automatically:
- Creates a Python virtual environment
- Installs all dependencies
- Creates `.env` from template

Then run:

```bash
source .venv/bin/activate
python main.py --dry-run
```

## Manual Setup (Alternative)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Usage

```bash
# Dry run (preview TOC only, no chapter writing)
python main.py --dry-run

# Full run (generates Markdown + PDF)
python main.py

# With content verification
python main.py --verify
```

## Output

| File | Description |
|---|---|
| `output/manuscript.md` | Full book in Markdown |
| `output/manuscript.pdf` | Formatted PDF |
| `output/chapters/` | Individual chapter files |
| `data/transcripts/` | Pre-fetched transcript data |
| `data/playlist_metadata/` | Playlist metadata & TOC |

## LLM Providers

| Provider | Model | API Key? | Speed |
|---|---|---|---|
| `gemini` (default) | `gemini-2.0-flash` | `GEMINI_API_KEY` | Fast |

To switch providers (if you extend the code), edit `config/default.yaml`:

```yaml
llm:
  primary_provider: gemini
```

Add your API keys to `.env`.

## Configuration

All settings in `config/default.yaml`:

| Setting | Description |
|---|---|
| `llm.primary_provider` | LLM for chapter writing (`gemini`) |
| `chunking.target_chunk_tokens` | Max tokens per chunk sent to LLM |
| `processing.filler_words` | Filler words/phrases to remove |
| `verification.enabled` | Enable verification pass |

## Project Structure

```
playlist_to_book/
├── setup.sh                       # One-command setup (installs everything)
├── main.py                        # CLI entry point & pipeline orchestrator
├── config/
│   └── default.yaml               # All pipeline settings
├── data/                          # Permanent playlist data (committed)
│   ├── transcripts/               # Pre-fetched transcript JSON files
│   └── playlist_metadata/         # Video metadata & TOC
├── src/
│   ├── config.py                  # Configuration loader
│   ├── transcript/
│   │   └── fetcher.py             # Data loader (reads from data/)
│   ├── processing/
│   │   ├── cleaner.py             # Filler removal, normalization
│   │   └── chunker.py             # Token-aware text splitting
│   ├── llm/
│   │   ├── base.py                # Abstract LLM provider
│   │   ├── gemini_provider.py     # Google Gemini (cloud)
│   │   └── factory.py             # Provider factory
│   ├── verification/
│   │   └── verifier.py            # Cross-LLM content verification
│   └── book/
│       ├── structurer.py          # TOC generation, chapter writing, refinement
│       └── exporter.py            # Markdown & PDF export
├── tests/
│   ├── test_cleaner.py
│   └── test_chunker.py
├── output/                        # Generated files (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## Pipeline Steps

1. **Load metadata** — read video metadata from `data/playlist_metadata/`
2. **Generate TOC** — LLM proposes chapter structure from video titles & summaries
3. **Write chapters** — load transcripts from `data/transcripts/`, clean, chunk, LLM → coherent chapters
4. **Export** — assemble Markdown manuscript and render PDF via WeasyPrint

## Requirements

- Python 3.10+
- API keys for Gemini

## License

MIT
