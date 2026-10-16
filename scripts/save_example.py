"""
Runs Signal on the sample reviews with your API key and saves the result as
the example report people can view without a key.

Run once from the project folder (with your key in .env):
    python scripts/save_example.py
"""

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import synthesizer as syn  # noqa: E402

load_dotenv(ROOT / ".env")
key = os.getenv("ANTHROPIC_API_KEY", "").strip()
if not key or key == "your-key-here":
    sys.exit("Add your Anthropic API key to .env first (see .env.example).")

df = pd.read_csv(ROOT / "sample_data" / "stride_reviews.csv")
context = "Stride, a running and fitness app with watch sync, guided training plans and a paid subscription."
report = syn.run(df, "review_text", key, syn.DEFAULT_MODEL, context,
                 on_step=lambda label, p: print(f"{p:4.0%}  {label}"))

note = f"Saved from a live run with {syn.DEFAULT_MODEL} on {date.today():%B %d, %Y}."
syn.save_report(report, str(ROOT / "sample_data" / "example_report.json"), note)
print("\nSaved sample_data/example_report.json")
print("Headline:", report.headline)
