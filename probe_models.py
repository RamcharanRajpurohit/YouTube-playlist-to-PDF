#!/usr/bin/env python3
"""
Probe available Gemini models and quota information for your account.

Usage:
    python probe_models.py

Requires GEMINI_API_KEY to be set in your .env file or environment.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

# ── Known rate limits per model (RPM = requests/min, TPM = tokens/min)
# Source: https://ai.google.dev/gemini-api/docs/rate-limits
_KNOWN_LIMITS: dict[str, dict] = {
    "gemini-2.0-flash": {
        "free_rpm": 15,
        "free_tpm": 1_000_000,
        "free_rpd": 1_500,
        "paid_rpm": 2_000,
        "paid_tpm": 4_000_000,
        "context_tokens": 1_048_576,
        "output_tokens": 8_192,
        "batch_support": True,
    },
    "gemini-2.0-flash-lite": {
        "free_rpm": 30,
        "free_tpm": 1_000_000,
        "free_rpd": 1_500,
        "paid_rpm": 4_000,
        "paid_tpm": 4_000_000,
        "context_tokens": 1_048_576,
        "output_tokens": 8_192,
        "batch_support": True,
    },
    "gemini-2.5-flash": {
        "free_rpm": 10,
        "free_tpm": 250_000,
        "free_rpd": 500,
        "paid_rpm": 1_000,
        "paid_tpm": 250_000,
        "context_tokens": 1_048_576,
        "output_tokens": 65_536,
        "batch_support": True,
    },
    "gemini-1.5-pro": {
        "free_rpm": 2,
        "free_tpm": 32_000,
        "free_rpd": 50,
        "paid_rpm": 1_000,
        "paid_tpm": 4_000_000,
        "context_tokens": 2_097_152,
        "output_tokens": 8_192,
        "batch_support": True,
    },
    "gemini-1.5-flash": {
        "free_rpm": 15,
        "free_tpm": 1_000_000,
        "free_rpd": 1_500,
        "paid_rpm": 2_000,
        "paid_tpm": 4_000_000,
        "context_tokens": 1_048_576,
        "output_tokens": 8_192,
        "batch_support": True,
    },
}

_PARALLEL_RECOMMENDATION: dict[str, int] = {
    "gemini-2.0-flash": 10,
    "gemini-2.0-flash-lite": 20,
    "gemini-2.5-flash": 5,
    "gemini-1.5-pro": 2,
    "gemini-1.5-flash": 10,
}


def probe_models() -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌  GEMINI_API_KEY not set. Please check your .env file.")
        return

    client = genai.Client(api_key=api_key)

    print("\n" + "═" * 62)
    print("  🔍  Gemini Model Probe")
    print("═" * 62)

    try:
        models = list(client.models.list())
    except Exception as exc:
        print(f"❌  Failed to list models: {exc}")
        return

    # Filter to text-generation models only
    text_models = [
        m for m in models
        if hasattr(m, "supported_actions") and "generateContent" in (m.supported_actions or [])
    ]

    if not text_models:
        # Fallback: show all models
        text_models = models

    print(f"\n  Found {len(text_models)} generative model(s) on your account:\n")

    for m in sorted(text_models, key=lambda x: x.name):
        model_id = m.name.replace("models/", "")
        display = getattr(m, "display_name", model_id)
        print(f"  ┌─ {display}")
        print(f"  │  ID:              {model_id}")

        if hasattr(m, "input_token_limit") and m.input_token_limit:
            print(f"  │  Context tokens:  {m.input_token_limit:,}")
        if hasattr(m, "output_token_limit") and m.output_token_limit:
            print(f"  │  Max output:      {m.output_token_limit:,}")

        # Lookup known rate limits
        known = None
        for key in _KNOWN_LIMITS:
            if key in model_id:
                known = _KNOWN_LIMITS[key]
                break

        if known:
            print(f"  │  ── Rate Limits ──────────────────────────")
            print(f"  │  Free tier:       {known['free_rpm']} RPM  /  {known['free_tpm']:,} TPM  /  {known['free_rpd']:,} RPD")
            print(f"  │  Paid tier:       {known['paid_rpm']} RPM  /  {known['paid_tpm']:,} TPM")
            print(f"  │  Batch API:       {'✅ Supported' if known['batch_support'] else '❌ Not supported'}")
            rec = _PARALLEL_RECOMMENDATION.get(model_id.split("-preview")[0].split("-exp")[0], 5)
            print(f"  │  Recommended parallel calls (paid): {rec}")
        else:
            print(f"  │  Rate limits: (not in local database — check ai.google.dev/gemini-api/docs/rate-limits)")

        print(f"  └{'─' * 50}")

    print()
    print("═" * 62)
    print("  📊  Recommendation for this pipeline (40 chapters):")
    print("═" * 62)
    print()
    print("  Model               │ Free safe batch │ Paid safe batch")
    print("  ────────────────────┼─────────────────┼────────────────")
    for model_id, limits in _KNOWN_LIMITS.items():
        free_safe = max(1, limits["free_rpm"] // 3)
        paid_safe = min(10, _PARALLEL_RECOMMENDATION.get(model_id, 5))
        print(f"  {model_id:<20}│ {free_safe:<15} │ {paid_safe}")

    print()
    print("  ℹ️   At paid-tier batch=10 for gemini-2.0-flash:")
    print("       40 chapters → ~4 batches → ~4× faster than sequential")
    print()


if __name__ == "__main__":
    probe_models()
