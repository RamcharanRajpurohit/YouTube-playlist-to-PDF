# Playlist → Book Pipeline

Convert any YouTube playlist (or single video) into a professionally formatted technical book manuscript, fully automated.

## Features

- **End-to-end pipeline**: playlist URL → transcripts → chapters → polished PDF
- **Works out of the box**: uses Ollama (local, free) by default — no API keys needed
- **Multi-LLM support**: Ollama (local), Gemini, Groq — easily swappable
- **Large content handling**: automatic transcript chunking with token-aware splitting
- **Caching & resumability**: transcripts and chapters are cached; a failed run can resume
- **Optional verification**: cross-check chapters against source transcripts using a second LLM
- **Book-quality output**: professional PDF with proper typography, code highlighting, and page layout

## Quick Start (One Command)

```bash
git clone <repo-url> && cd playlist_to_book
chmod +x setup.sh && ./setup.sh
```

This automatically:
- Creates a Python virtual environment
- Installs all dependencies
- Installs Ollama (if not present)
- Pulls the Mistral model
- Creates `.env` from template

Then run:

```bash
source .venv/bin/activate
python main.py --url "https://youtube.com/watch?v=Xpr8D6LeAtw&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu" --dry-run
```

## Manual Setup (Alternative)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Install Ollama: https://ollama.com/download
ollama pull mistral
```

## Usage

```bash
# Dry run (fetch transcripts + preview TOC, no LLM calls)
python main.py --url "<playlist_or_video_url>" --dry-run

# Full run (generates Markdown + PDF)
python main.py --url "<playlist_or_video_url>"

# With content verification
python main.py --url "<url>" --verify

# Custom title
python main.py --url "<url>" --title "My Custom Book"
```

## Output

| File | Description |
|---|---|
| `output/manuscript.md` | Full book in Markdown |
| `output/manuscript.pdf` | Formatted PDF |
| `output/chapters/` | Individual chapter files |
| `output/transcripts/` | Cached raw transcripts |

## LLM Providers

| Provider | Model | API Key? | Speed |
|---|---|---|---|
| `ollama` (default) | `mistral` | **None** | Slower (local CPU) |
| `gemini` | `gemini-2.5-flash` | `GEMINI_API_KEY` | Fast |
| `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | Fast |

To switch providers, edit `config/default.yaml`:

```yaml
llm:
  primary_provider: gemini   # or groq, ollama
```

For cloud providers, add API keys to `.env`.

## Configuration

All settings in `config/default.yaml`:

| Setting | Description |
|---|---|
| `llm.primary_provider` | LLM for chapter writing (`ollama`, `gemini`, `groq`) |
| `llm.secondary_provider` | LLM for verification |
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
├── src/
│   ├── config.py                  # Configuration loader
│   ├── transcript/
│   │   └── fetcher.py             # Playlist extraction & transcript fetching
│   ├── processing/
│   │   ├── cleaner.py             # Filler removal, normalization
│   │   └── chunker.py             # Token-aware text splitting
│   ├── llm/
│   │   ├── base.py                # Abstract LLM provider
│   │   ├── gemini_provider.py     # Google Gemini (cloud)
│   │   ├── groq_provider.py       # Groq (cloud)
│   │   ├── ollama_provider.py     # Ollama (local, free)
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

1. **Extract videos** — resolve playlist URL → video metadata via `yt-dlp`
2. **Fetch transcripts** — download/cache transcripts via `youtube-transcript-api`
3. **Clean transcripts** — remove filler words, normalize text
4. **Generate TOC** — LLM proposes chapter structure from video titles & summaries
5. **Write chapters** — transcript chunks → LLM → merge into coherent chapters → refinement
6. **Verify** *(optional)* — secondary LLM cross-checks for hallucinations
7. **Export** — assemble Markdown manuscript and render PDF via WeasyPrint

## Requirements

- Python 3.10+
- For local mode: Ollama (auto-installed by `setup.sh`)
- For cloud mode: API keys for Gemini and/or Groq

## License

MIT
