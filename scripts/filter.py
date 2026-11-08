"""
paper-filter: Semantic RSS filter for academic preprints.

Fetches configured RSS feeds, scores each entry against an interest profile
(derived from a Zotero export or a plain-text description), and outputs a
filtered RSS feed for subscription in any RSS reader (e.g. Inoreader).
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
import requests
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
# Interest profile
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def profile_from_description(description: str, model: SentenceTransformer) -> np.ndarray:
    """Embed a plain-text interest description."""
    log.info("Building interest profile from text description.")
    vec = model.encode([description], normalize_embeddings=True)
    return vec[0]


def profile_from_bibtex(bib_path: str, model: SentenceTransformer) -> np.ndarray:
    """Extract titles+abstracts from a BibTeX file and average their embeddings."""
    log.info(f"Building interest profile from BibTeX: {bib_path}")
    texts = []
    current_title = current_abstract = ""
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
    log.info(f"  Found {len(texts)} entries in BibTeX.")
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    return vecs.mean(axis=0)


def profile_from_csv(csv_path: str, model: SentenceTransformer) -> np.ndarray:
    """Extract titles+abstracts from a Zotero CSV export and average their embeddings."""
    log.info(f"Building interest profile from CSV: {csv_path}")
    texts = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("Title", "") or row.get("title", "")
            abstract = row.get("Abstract Note", "") or row.get("abstract", "")
            if title:
                texts.append(f"{title}. {abstract}".strip())
    if not texts:
        raise ValueError("No entries found in CSV file.")
    log.info(f"  Found {len(texts)} entries in CSV.")
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    return vecs.mean(axis=0)


def profile_from_ris(ris_path: str, model: SentenceTransformer) -> np.ndarray:
    """Extract titles+abstracts from a RIS/Zotero .txt export and average their embeddings.
    RIS format uses tags like %T (title) and %X (abstract).
    """
    log.info(f"Building interest profile from RIS: {ris_path}")
    texts = []
    current_title = ""
    current_abstract = ""
    with open(ris_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("%T "):
                current_title = line[3:].strip()
            elif line.startswith("%X "):
                current_abstract = line[3:].strip()
            elif line.startswith("%X") and len(line) > 2:
                # continuation lines for long abstracts
                current_abstract += " " + line[2:].strip()
            elif line == "" and current_title:
                texts.append(f"{current_title}. {current_abstract}".strip())
                current_title = ""
                current_abstract = ""
    # catch last entry if file doesn't end with blank line
    if current_title:
        texts.append(f"{current_title}. {current_abstract}".strip())
    if not texts:
        raise ValueError("No entries found in RIS file.")
    log.info(f"  Found {len(texts)} entries in RIS export.")
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    return vecs.mean(axis=0)


def build_interest_profile(cfg: dict, model: SentenceTransformer) -> np.ndarray:
    zotero_path = cfg.get("zotero_export")
    if zotero_path and Path(zotero_path).exists():
        if zotero_path.endswith(".bib"):
            return profile_from_bibtex(zotero_path, model)
        elif zotero_path.endswith(".csv"):
            return profile_from_csv(zotero_path, model)
        elif zotero_path.endswith(".txt") or zotero_path.endswith(".ris"):
            return profile_from_ris(zotero_path, model)
        else:
            log.warning(f"Unknown Zotero export format: {zotero_path}. Falling back to description.")
    desc = cfg.get("interest_description", "")
    if not desc.strip():
        raise ValueError("No interest profile configured. Set zotero_export or interest_description in config.")
    return profile_from_description(desc, model)

# ---------------------------------------------------------------------------
# Feed fetching
# ---------------------------------------------------------------------------

def fetch_feed(feed_cfg: dict, max_age_days: int) -> list[dict]:
    """Fetch and parse one RSS feed, returning a list of entry dicts."""
    name = feed_cfg["name"]
    url = feed_cfg["url"]
    log.info(f"Fetching: {name}")
    try:
        parsed = feedparser.parse(url, request_headers={"User-Agent": "paper-filter/1.0"})
    except Exception as e:
        log.warning(f"  Failed to fetch {name}: {e}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    entries = []
    for e in parsed.entries:
        # Parse date
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
# Scoring
# ---------------------------------------------------------------------------

def score_entries(entries: list[dict], profile: np.ndarray, model: SentenceTransformer) -> list[dict]:
    if not entries:
        return []
    texts = [e["text"] for e in entries]
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    scores = (vecs @ profile).tolist()
    for e, s in zip(entries, scores):
        e["score"] = round(float(s), 4)
    return entries

# ---------------------------------------------------------------------------
# RSS output
# ---------------------------------------------------------------------------

def build_rss(entries: list[dict], cfg: dict) -> str:
    out_cfg = cfg.get("output", {})
    title = out_cfg.get("feed_title", "Filtered Paper Feed")
    desc = out_cfg.get("feed_description", "Semantically filtered papers")

    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = title
    ET.SubElement(channel, "description").text = desc
    ET.SubElement(channel, "link").text = "https://github.com"
    ET.SubElement(channel, "lastBuildDate").text = datetime.now(timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S +0000"
    )

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
    parser = argparse.ArgumentParser(description="Semantic paper feed filter")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, don't write output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    threshold = cfg.get("threshold", 0.25)
    top_n = cfg.get("top_n", 10)
    max_age_days = cfg.get("output", {}).get("max_age_days", 7)
    passthrough = set(cfg.get("passthrough_feeds", []))
    out_cfg = cfg.get("output", {})
    out_file = out_cfg.get("filename", "filtered_feed.xml")

    # Load model (cached after first run)
    log.info("Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build interest profile
    profile = build_interest_profile(cfg, model)
    log.info("Interest profile ready.")

    # Fetch and filter per-feed
    filtered = []
    passthrough_entries = []
    total_fetched = 0
    total_above_threshold = 0

    for feed_cfg in cfg.get("feeds", []):
        entries = fetch_feed(feed_cfg, max_age_days)
        total_fetched += len(entries)

        if feed_cfg["name"] in passthrough:
            passthrough_entries.extend(entries)
            continue

        # Deduplicate within this feed by title
        seen = set()
        unique = []
        for e in entries:
            key = e["title"].lower()[:80]
            if key not in seen:
                seen.add(key)
                unique.append(e)

        if not unique:
            continue

        # Score this feed independently
        scored = score_entries(unique, profile, model)
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Per-feed filter: above threshold + always keep top_n
        above = [e for e in scored if e["score"] >= threshold]
        total_above_threshold += len(above)

        # Always include top_n from this feed as a floor
        top = scored[:top_n]
        combined = {e["id"]: e for e in above}
        for e in top:
            combined[e["id"]] = e

        feed_filtered = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        log.info(f"  {feed_cfg['name']}: {len(unique)} unique → {len(above)} above threshold → {len(feed_filtered)} in feed")
        filtered.extend(feed_filtered)

    # Sort all filtered entries by score, newest within same score
    filtered.sort(key=lambda x: x["score"], reverse=True)

    # Add passthrough entries (journals) at the end, unscored
    for e in passthrough_entries:
        e["score"] = 1.0  # mark as passthrough
        filtered.append(e)

    log.info(f"Total fetched: {total_fetched} | Above threshold: {total_above_threshold} | In feed: {len(filtered)}")

    if args.dry_run:
        log.info("Dry run — top 10 results:")
        for e in filtered[:10]:
            log.info(f"  [{e['score']:.3f}] {e['title'][:80]}")
        return

    rss_content = build_rss(filtered, cfg)
    Path(out_file).write_text(rss_content, encoding="utf-8")
    log.info(f"Written: {out_file} ({len(filtered)} items)")

    # Write a small JSON summary for debugging
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_fetched": total_fetched,
        "above_threshold": total_above_threshold,
        "in_feed": len(filtered),
        "threshold": threshold,
        "top_n_per_feed": top_n,
    }
    Path(out_file.replace(".xml", "_summary.json")).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
