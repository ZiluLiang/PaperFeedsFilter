"""
paper-filter: Semantic RSS filter for academic preprints.

Produces two RSS feeds:
  - Core feed:    high threshold, scored against Zotero library profile
  - Explore feed: lower threshold, scored against exploration description
"""

import os
import re
import sys
import csv
import json
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import yaml
import numpy as np
import feedparser
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()

# ---------------------------------------------------------------------------
# Interest profiles
# ---------------------------------------------------------------------------

def profile_from_description(description: str, model: SentenceTransformer) -> np.ndarray:
    log.info("Building profile from text description.")
    return model.encode([description], normalize_embeddings=True)[0]


def profile_from_bibtex(bib_path: str, model: SentenceTransformer) -> np.ndarray:
    log.info(f"Building profile from BibTeX: {bib_path}")
    texts, current_title, current_abstract = [], "", ""
    with open(bib_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("title"):
                current_title = re.sub(r"[{}]", "", line.split("=", 1)[-1].strip().strip(","))
            elif line.lower().startswith("abstract"):
                current_abstract = re.sub(r"[{}]", "", line.split("=", 1)[-1].strip().strip(","))
            elif line == "}":
                if current_title:
                    texts.append(f"{current_title}. {current_abstract}".strip())
                current_title = current_abstract = ""
    if not texts:
        raise ValueError("No entries found in BibTeX file.")
    log.info(f"  {len(texts)} entries found.")
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    return vecs.mean(axis=0)


def profile_from_csv(csv_path: str, model: SentenceTransformer) -> np.ndarray:
    log.info(f"Building profile from CSV: {csv_path}")
    texts = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = row.get("Title", "") or row.get("title", "")
            abstract = row.get("Abstract Note", "") or row.get("abstract", "")
            if title:
                texts.append(f"{title}. {abstract}".strip())
    if not texts:
        raise ValueError("No entries found in CSV file.")
    log.info(f"  {len(texts)} entries found.")
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    return vecs.mean(axis=0)


def profile_from_ris(ris_path: str, model: SentenceTransformer) -> np.ndarray:
    log.info(f"Building profile from RIS: {ris_path}")
    texts, current_title, current_abstract = [], "", ""
    with open(ris_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("%T "):
                current_title = line[3:].strip()
            elif line.startswith("%X "):
                current_abstract = line[3:].strip()
            elif line == "" and current_title:
                texts.append(f"{current_title}. {current_abstract}".strip())
                current_title = current_abstract = ""
    if current_title:
        texts.append(f"{current_title}. {current_abstract}".strip())
    if not texts:
        raise ValueError("No entries found in RIS file.")
    log.info(f"  {len(texts)} entries found.")
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    return vecs.mean(axis=0)


def build_zotero_profile(cfg: dict, model: SentenceTransformer) -> np.ndarray:
    path = cfg.get("zotero_export")
    if path and Path(path).exists():
        if path.endswith(".bib"):
            return profile_from_bibtex(path, model)
        elif path.endswith(".csv"):
            return profile_from_csv(path, model)
        elif path.endswith(".txt") or path.endswith(".ris"):
            return profile_from_ris(path, model)
        else:
            log.warning(f"Unknown format: {path}. Falling back to description.")
    desc = cfg.get("interest_description", "")
    if not desc.strip():
        raise ValueError("No interest profile configured.")
    return profile_from_description(desc, model)


def generate_exploration_description(cfg: dict) -> str:
    """Use Claude API to infer adjacent interests from Zotero library.
    Saves the result to config/exploration_description.txt so it only runs once
    (delete that file to regenerate).
    """
    import requests

    cache_path = Path("config/exploration_description.txt")
    if cache_path.exists():
        log.info("Loading cached exploration description.")
        return cache_path.read_text(encoding="utf-8").strip()

    log.info("Generating exploration description from Zotero library via Claude API...")

    # Load titles from Zotero
    zotero_path = cfg.get("zotero_export", "")
    titles = []
    if zotero_path and Path(zotero_path).exists():
        with open(zotero_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("%T "):
                    titles.append(line[3:].strip())

    if not titles:
        raise ValueError("Could not read titles from Zotero export.")

    # Sample up to 200 titles to keep prompt size reasonable
    import random
    random.seed(42)
    sample = random.sample(titles, min(200, len(titles)))
    titles_text = "\n".join(f"- {t}" for t in sample)

    prompt = f"""Here are a sample of {len(sample)} paper titles from a researcher's personal library:

{titles_text}

Based on these titles, please:
1. Briefly characterise this researcher's core interests in 2-3 sentences.
2. Then write a 4-6 sentence description of ADJACENT fields and topics they don't currently read much about, but would likely find intellectually interesting given their existing interests. Focus on areas that would genuinely surprise and expand their reading — not just more of the same.

Write only the adjacent fields description (point 2), in plain prose, as if describing what a curious version of this researcher might want to explore next. Do not use bullet points or headers."""

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"Content-Type": "application/json"},
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=60
    )
    response.raise_for_status()
    data = response.json()
    description = data["content"][0]["text"].strip()

    log.info(f"Generated exploration description:\n{description}")
    cache_path.write_text(description, encoding="utf-8")
    log.info(f"Saved to {cache_path}")
    return description


def build_explore_profile(cfg: dict, model: SentenceTransformer) -> np.ndarray:
    # Use manual description if provided, otherwise auto-generate from Zotero
    desc = cfg.get("exploration_description", "").strip()
    if not desc:
        desc = generate_exploration_description(cfg)
    return profile_from_description(desc, model)

# ---------------------------------------------------------------------------
# Feed fetching
# ---------------------------------------------------------------------------

def fetch_feed(feed_cfg: dict, max_age_days: int) -> list[dict]:
    name = feed_cfg["name"]
    url = feed_cfg["url"]
    log.info(f"Fetching: {name}")
    try:
        parsed = feedparser.parse(url, request_headers={"User-Agent": "paper-filter/1.0"})
    except Exception as e:
        log.warning(f"  Failed: {e}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    entries = []
    for e in parsed.entries:
        pub = None
        for field in ("published", "updated"):
            raw = e.get(field)
            if raw:
                try:
                    pub = parsedate_to_datetime(raw)
                    if pub.tzinfo is None:
                        pub = pub.replace(tzinfo=timezone.utc)
                    break
                except Exception:
                    pass
        if pub and pub < cutoff:
            continue

        title = _clean(e.get("title", ""))
        abstract = _clean(e.get("summary", "") or e.get("description", ""))
        link = e.get("link", "")
        entry_id = e.get("id", link) or hashlib.md5(title.encode()).hexdigest()

        entries.append({
            "id": entry_id,
            "title": title,
            "abstract": abstract,
            "link": link,
            "published": pub.isoformat() if pub else "",
            "source": name,
            "text": f"{title}. {abstract}",
        })

    log.info(f"  {len(entries)} entries (within {max_age_days} days)")
    return entries

# ---------------------------------------------------------------------------
# Scoring & filtering
# ---------------------------------------------------------------------------

def score_entries(entries: list[dict], profile: np.ndarray, model: SentenceTransformer) -> list[dict]:
    if not entries:
        return []
    vecs = model.encode([e["text"] for e in entries], normalize_embeddings=True,
                        batch_size=64, show_progress_bar=False)
    scores = (vecs @ profile).tolist()
    for e, s in zip(entries, scores):
        e["score"] = round(float(s), 4)
    return entries


def filter_feed_entries(feed_cfg: dict, max_age_days: int, profile: np.ndarray,
                        model: SentenceTransformer, threshold: float, top_n: int) -> list[dict]:
    entries = fetch_feed(feed_cfg, max_age_days)
    if not entries:
        return []

    # Deduplicate by title
    seen, unique = set(), []
    for e in entries:
        key = e["title"].lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(e)

    scored = score_entries(unique, profile, model)
    scored.sort(key=lambda x: x["score"], reverse=True)

    above = [e for e in scored if e["score"] >= threshold]
    combined = {e["id"]: e for e in above}
    for e in scored[:top_n]:
        combined[e["id"]] = e

    result = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    log.info(f"  → {len(unique)} unique | {len(above)} above threshold | {len(result)} in output")
    return result

# ---------------------------------------------------------------------------
# RSS output
# ---------------------------------------------------------------------------

def build_rss(entries: list[dict], title: str, description: str) -> str:
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = title
    ET.SubElement(channel, "description").text = description
    ET.SubElement(channel, "link").text = "https://github.com"
    ET.SubElement(channel, "lastBuildDate").text = datetime.now(timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S +0000")

    for e in entries:
        item = ET.SubElement(channel, "item")
        score_pct = int(e["score"] * 100)
        ET.SubElement(item, "title").text = f"[{score_pct}%] [{e['source']}] {e['title']}"
        ET.SubElement(item, "link").text = e["link"]
        ET.SubElement(item, "guid").text = e["id"]
        body = e["abstract"] or "(no abstract)"
        ET.SubElement(item, "description").text = (
            f"<b>Relevance: {score_pct}%</b> | <b>Source:</b> {e['source']}<br/><br/>{body}"
        )
        if e["published"]:
            ET.SubElement(item, "pubDate").text = e["published"]

    ET.indent(rss, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(rss, encoding="unicode")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    max_age_days = cfg.get("output", {}).get("max_age_days", 7)
    core_threshold = cfg.get("core_threshold", 0.30)
    explore_threshold = cfg.get("explore_threshold", 0.18)
    top_n = cfg.get("top_n", 5)
    out = cfg.get("output", {})

    log.info("Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    log.info("=== Building core profile (Zotero) ===")
    core_profile = build_zotero_profile(cfg, model)

    log.info("=== Building explore profile (description) ===")
    explore_profile = build_explore_profile(cfg, model)

    # --- Core feed ---
    log.info("=== Processing core feeds ===")
    core_entries = []
    for feed_cfg in cfg.get("core_feeds", []):
        entries = filter_feed_entries(feed_cfg, max_age_days, core_profile,
                                      model, core_threshold, top_n)
        core_entries.extend(entries)

    # Deduplicate across feeds by id, re-sort
    seen = {}
    for e in core_entries:
        if e["id"] not in seen or e["score"] > seen[e["id"]]["score"]:
            seen[e["id"]] = e
    core_entries = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    # --- Explore feed ---
    log.info("=== Processing explore feeds ===")
    explore_entries = []
    for feed_cfg in cfg.get("explore_feeds", []):
        entries = filter_feed_entries(feed_cfg, max_age_days, explore_profile,
                                      model, explore_threshold, top_n)
        explore_entries.extend(entries)

    # Deduplicate, exclude anything already in core feed
    core_ids = {e["id"] for e in core_entries}
    seen = {}
    for e in explore_entries:
        if e["id"] not in core_ids and (e["id"] not in seen or e["score"] > seen[e["id"]]["score"]):
            seen[e["id"]] = e
    explore_entries = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    log.info(f"Core feed: {len(core_entries)} entries")
    log.info(f"Explore feed: {len(explore_entries)} entries")

    if args.dry_run:
        log.info("Core top 5:")
        for e in core_entries[:5]:
            log.info(f"  [{e['score']:.3f}] {e['title'][:80]}")
        log.info("Explore top 5:")
        for e in explore_entries[:5]:
            log.info(f"  [{e['score']:.3f}] {e['title'][:80]}")
        return

    # Write feeds
    core_file = out.get("core_filename", "filtered_feed.xml")
    explore_file = out.get("explore_filename", "explore_feed.xml")

    Path(core_file).write_text(
        build_rss(core_entries, out.get("core_title", "Core Feed"),
                  out.get("core_description", "")), encoding="utf-8")
    log.info(f"Written: {core_file}")

    Path(explore_file).write_text(
        build_rss(explore_entries, out.get("explore_title", "Explore Feed"),
                  out.get("explore_description", "")), encoding="utf-8")
    log.info(f"Written: {explore_file}")

    Path("feed_summary.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "core_entries": len(core_entries),
        "explore_entries": len(explore_entries),
        "core_threshold": core_threshold,
        "explore_threshold": explore_threshold,
    }, indent=2))


if __name__ == "__main__":
    main()
