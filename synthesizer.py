"""
The engine behind Signal. It turns a pile of raw feedback into a ranked report
in four plain steps:

    1. Discover  - the AI reads a sample of the feedback and proposes 4–8 themes.
    2. Tag       - the AI labels every comment with one theme and a sentiment.
    3. Rank      - plain Python counts mentions and orders the themes.
    4. Summarize - the AI writes a short executive summary of the results.

Only steps 1, 2 and 4 use AI. Step 3 is ordinary counting, so the numbers you
see in the report are exact, not guessed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_ROWS = 500          # safety cap so a large upload can't run up a big bill
BATCH_SIZE = 40         # comments sent to the AI per request when tagging
DISCOVERY_SAMPLE = 200  # comments the AI reads to propose themes
MAX_CHARS = 600         # long comments are trimmed to this length

OTHER = "Other"
SENTIMENTS = ["positive", "neutral", "negative"]
KINDS = ["problem", "request", "praise"]


# --------------------------------------------------------------------------- #
#  Data shapes
# --------------------------------------------------------------------------- #
@dataclass
class Theme:
    name: str
    description: str
    kind: str                     # problem | request | praise
    mentions: int = 0
    share: float = 0.0            # % of all comments
    negative: int = 0
    positive: int = 0
    neutral: int = 0
    quotes: list[str] = field(default_factory=list)


@dataclass
class Report:
    themes: list[Theme]
    tagged: pd.DataFrame          # original rows + theme + sentiment columns
    headline: str
    summary: str
    recommendation: str
    total: int
    truncated: bool               # True if rows were cut to MAX_ROWS

    def by_kind(self, kind: str) -> list[Theme]:
        return [t for t in self.themes if t.kind == kind]


# --------------------------------------------------------------------------- #
#  Talking to the AI
# --------------------------------------------------------------------------- #
def _ask_json(client: Anthropic, model: str, system: str, prompt: str,
              schema: dict, max_tokens: int = 4000) -> dict:
    """Ask Claude a question and get a structured (JSON) answer back.

    We describe the answer's shape as a 'tool' and require Claude to use it,
    which reliably returns clean JSON instead of free text.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[{"name": "answer", "description": "Return the answer.", "input_schema": schema}],
        tool_choice={"type": "tool", "name": "answer"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("The AI did not return a structured answer.")


def _clean(text: str) -> str:
    return " ".join(str(text).split())[:MAX_CHARS]


def _distinct(texts: list[str], limit: int) -> list[str]:
    """Pick up to `limit` texts, skipping near-repeats of one already picked
    (more than half the words of the shorter text are shared)."""
    picked, seen = [], []
    for text in texts:
        words = set(text.lower().split())
        if all(len(words & s) / max(min(len(words), len(s)), 1) <= 0.5 for s in seen):
            picked.append(text)
            seen.append(words)
        if len(picked) == limit:
            break
    return picked


def _numbered(texts: list[str], start: int = 0) -> str:
    return "\n".join(f"[{start + i}] {t}" for i, t in enumerate(texts))


SYSTEM = (
    "You are a senior product manager's research assistant. You analyze raw "
    "customer feedback carefully and literally: you only report what users "
    "actually say, and you never invent problems that aren't in the text."
)


# --------------------------------------------------------------------------- #
#  Step 1 — Discover themes
# --------------------------------------------------------------------------- #
def discover_themes(client: Anthropic, model: str, texts: list[str],
                    context: str = "") -> list[Theme]:
    sample = texts if len(texts) <= DISCOVERY_SAMPLE else random.Random(0).sample(texts, DISCOVERY_SAMPLE)
    schema = {
        "type": "object",
        "properties": {
            "themes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "2–5 word theme name, plain language"},
                        "description": {"type": "string", "description": "One sentence on what users say"},
                        "kind": {"type": "string", "enum": KINDS},
                    },
                    "required": ["name", "description", "kind"],
                },
            }
        },
        "required": ["themes"],
    }
    prompt = (
        f"{'Product context: ' + context + chr(10) + chr(10) if context else ''}"
        "Read the customer feedback below and propose between 4 and 8 themes "
        "that together cover most of it. Each theme should be specific and "
        "actionable (e.g. 'Watch sync failures', not 'Technical issues'). "
        "Mark each theme as a 'problem' (something broken or frustrating), a "
        "'request' (something users want added) or 'praise' (something users love). "
        "Don't create a catch-all theme; leftovers will go into 'Other'.\n\n"
        f"FEEDBACK:\n{_numbered(sample)}"
    )
    data = _ask_json(client, model, SYSTEM, prompt, schema)
    themes, seen = [], set()
    for t in data.get("themes", []):
        name = t.get("name", "").strip()
        if name and name.lower() not in seen and name != OTHER:
            seen.add(name.lower())
            kind = t.get("kind") if t.get("kind") in KINDS else "problem"
            themes.append(Theme(name=name, description=t.get("description", ""), kind=kind))
    if not themes:
        raise RuntimeError("The AI couldn't find any themes in this feedback.")
    return themes


# --------------------------------------------------------------------------- #
#  Step 2 — Tag every comment
# --------------------------------------------------------------------------- #
def tag_feedback(client: Anthropic, model: str, texts: list[str], themes: list[Theme],
                 on_progress: Callable[[float], None] | None = None) -> list[dict]:
    names = [t.name for t in themes] + [OTHER]
    theme_list = "\n".join(f"- {t.name}: {t.description}" for t in themes)
    schema = {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "theme": {"type": "string", "enum": names},
                        "sentiment": {"type": "string", "enum": SENTIMENTS},
                    },
                    "required": ["id", "theme", "sentiment"],
                },
            }
        },
        "required": ["labels"],
    }

    labels: dict[int, dict] = {}
    batches = range(0, len(texts), BATCH_SIZE)
    for n, start in enumerate(batches, 1):
        chunk = texts[start:start + BATCH_SIZE]
        prompt = (
            f"Themes:\n{theme_list}\n- {OTHER}: doesn't fit any theme above\n\n"
            "For EVERY numbered comment below, pick the single best theme and the "
            "comment's sentiment. Return one label per comment, using its number as id.\n\n"
            f"COMMENTS:\n{_numbered(chunk, start)}"
        )
        data = _ask_json(client, model, SYSTEM, prompt, schema, max_tokens=8000)
        for item in data.get("labels", []):
            i = item.get("id")
            if isinstance(i, int) and start <= i < start + len(chunk):
                labels[i] = item
        if on_progress:
            on_progress(n / len(batches))

    # Anything the AI skipped or mislabelled is filed safely under "Other".
    result = []
    for i in range(len(texts)):
        item = labels.get(i, {})
        theme = item.get("theme") if item.get("theme") in names else OTHER
        sentiment = item.get("sentiment") if item.get("sentiment") in SENTIMENTS else "neutral"
        result.append({"theme": theme, "sentiment": sentiment})
    return result


# --------------------------------------------------------------------------- #
#  Step 3 — Rank (plain counting, no AI)
# --------------------------------------------------------------------------- #
def rank_themes(tagged: pd.DataFrame, text_col: str, themes: list[Theme]) -> list[Theme]:
    total = len(tagged)
    want = {"problem": "negative", "request": None, "praise": "positive"}
    for t in themes:
        rows = tagged[tagged["theme"] == t.name]
        counts = rows["sentiment"].value_counts()
        t.mentions = len(rows)
        t.share = round(100 * t.mentions / total, 1) if total else 0.0
        t.negative = int(counts.get("negative", 0))
        t.positive = int(counts.get("positive", 0))
        t.neutral = int(counts.get("neutral", 0))

        # Quotes: prefer comments whose tone matches the theme, then the most
        # descriptive (longest) ones, skipping duplicates.
        pool = rows
        if want[t.kind]:
            matching = rows[rows["sentiment"] == want[t.kind]]
            pool = matching if len(matching) else rows
        ordered = pool[text_col].astype(str).sort_values(key=lambda s: s.str.len(), ascending=False)
        t.quotes = _distinct([_clean(q) for q in ordered], limit=3)

    themes = [t for t in themes if t.mentions > 0]
    # Problems are ordered by how many unhappy users mention them;
    # requests and praise simply by how often they come up.
    return sorted(themes, key=lambda t: (KINDS.index(t.kind),
                                         -(t.negative if t.kind == "problem" else t.mentions)))


# --------------------------------------------------------------------------- #
#  Step 4 — Executive summary
# --------------------------------------------------------------------------- #
def summarize(client: Anthropic, model: str, themes: list[Theme], total: int,
              context: str = "") -> dict:
    table = "\n".join(
        f"- [{t.kind}] {t.name}: {t.mentions} mentions ({t.share}%), "
        f"{t.negative} negative, {t.positive} positive. {t.description}"
        for t in themes
    )
    schema = {
        "type": "object",
        "properties": {
            "headline": {"type": "string", "description": "One sentence, max 15 words, the single biggest takeaway"},
            "summary": {"type": "string", "description": "2–3 sentences for a busy executive"},
            "recommendation": {"type": "string", "description": "One sentence: if the team fixes one thing next, what and why"},
        },
        "required": ["headline", "summary", "recommendation"],
    }
    prompt = (
        f"{'Product context: ' + context + chr(10) + chr(10) if context else ''}"
        f"We analyzed {total} customer comments. Here are the themes with exact counts:\n"
        f"{table}\n\nWrite a crisp, specific executive summary. Use the numbers. No fluff."
    )
    return _ask_json(client, model, SYSTEM, prompt, schema, max_tokens=1000)


# --------------------------------------------------------------------------- #
#  The whole pipeline
# --------------------------------------------------------------------------- #
def run(df: pd.DataFrame, text_col: str, api_key: str, model: str = DEFAULT_MODEL,
        context: str = "", on_step: Callable[[str, float], None] | None = None) -> Report:
    """Run all four steps. `on_step(label, progress)` lets the UI show progress."""
    step = on_step or (lambda *_: None)
    client = Anthropic(api_key=api_key)

    data = df.copy()
    data = data[data[text_col].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    truncated = len(data) > MAX_ROWS
    data = data.head(MAX_ROWS)
    if data.empty:
        raise ValueError("That column has no text in it.")
    texts = [_clean(t) for t in data[text_col]]

    step("Reading the feedback and finding themes", 0.05)
    themes = discover_themes(client, model, texts, context)

    step("Tagging every comment", 0.2)
    tags = tag_feedback(client, model, texts, themes,
                        on_progress=lambda p: step("Tagging every comment", 0.2 + 0.6 * p))
    data["theme"] = [t["theme"] for t in tags]
    data["sentiment"] = [t["sentiment"] for t in tags]

    step("Counting and ranking", 0.85)
    ranked = rank_themes(data, text_col, themes)

    step("Writing the summary", 0.9)
    s = summarize(client, model, ranked, len(data), context)

    step("Done", 1.0)
    return Report(themes=ranked, tagged=data, headline=s.get("headline", ""),
                  summary=s.get("summary", ""), recommendation=s.get("recommendation", ""),
                  total=len(data), truncated=truncated)


# --------------------------------------------------------------------------- #
#  Export
# --------------------------------------------------------------------------- #
def to_markdown(report: Report, title: str = "Feedback report") -> str:
    out = [f"# {title}", "", f"**{report.headline}**", "", report.summary, "",
           f"> **Recommendation:** {report.recommendation}", "",
           f"_Based on {report.total} comments. Analyzed with Signal._", ""]
    sections = [("problem", "Top problems"), ("request", "Feature requests"), ("praise", "What users love")]
    for kind, heading in sections:
        themes = report.by_kind(kind)
        if not themes:
            continue
        out += [f"## {heading}", ""]
        for i, t in enumerate(themes, 1):
            out += [f"### {i}. {t.name}",
                    f"{t.mentions} mentions ({t.share}%) · {t.negative} negative · {t.positive} positive",
                    "", t.description, ""]
            out += [f"> “{q}”" for q in t.quotes] + [""]
    other = int((report.tagged["theme"] == OTHER).sum())
    if other:
        out += [f"_{other} comments didn't fit any theme and were filed under “Other”._", ""]
    return "\n".join(out)



# --------------------------------------------------------------------------- #
#  Save / load a report (used for the no-key example report)
# --------------------------------------------------------------------------- #
def save_report(report: Report, path: str, note: str = "") -> None:
    data = {
        "note": note,
        "headline": report.headline, "summary": report.summary,
        "recommendation": report.recommendation, "total": report.total,
        "truncated": report.truncated,
        "themes": [t.__dict__ for t in report.themes],
        "tagged": report.tagged.to_dict(orient="records"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False, default=str)


def load_report(path: str) -> tuple[Report, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    report = Report(
        themes=[Theme(**t) for t in data["themes"]],
        tagged=pd.DataFrame(data["tagged"]),
        headline=data["headline"], summary=data["summary"],
        recommendation=data["recommendation"], total=data["total"],
        truncated=data.get("truncated", False),
    )
    return report, data.get("note", "")
