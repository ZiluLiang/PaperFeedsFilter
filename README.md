# paper-filter

Semantic RSS filter for academic preprints. Scores incoming papers against your
interest profile (derived from your Zotero library or a plain-text description)
and publishes a filtered RSS feed you can subscribe to in Inoreader or any
other reader.

Runs free on GitHub Actions + GitHub Pages. No API keys needed.

---

## Setup (one-time, ~10 minutes)

### 1. Fork / create the repo

Click **Use this template** (or fork). Make the repo **public** — GitHub Pages
requires this on the free plan.

### 2. Enable GitHub Pages

Go to **Settings → Pages → Source** and set it to the `gh-pages` branch, root
folder. Save.

### 3. Configure your feeds and interests

Edit `config/config.yaml`:

- **Add or remove RSS feeds** under the `feeds:` section.
- **Set your interest profile** — either point `zotero_export` at your library
  file (see below), or edit the `interest_description` text.
- **Tune the threshold** — start at `0.25`. Raise it if too many papers come
  through; lower it if too few.

### 4. (Optional) Add your Zotero library

Export from Zotero: **File → Export Library → BibTeX** (recommended) or CSV.

Place the exported file in the `config/` folder (e.g. `config/my_library.bib`),
then update `config.yaml`:

```yaml
zotero_export: "config/my_library.bib"
```

Commit and push. The filter will use your library to build your interest profile.
To refresh it later, just re-export and replace the file.

### 5. Run the workflow

Go to **Actions → Filter Papers → Run workflow**. Wait ~2 minutes.

The filtered feed will be published to:
```
https://<your-username>.github.io/<repo-name>/filtered_feed.xml
```

### 6. Subscribe in Inoreader

Copy the URL above and add it as a new feed in Inoreader. Done — it updates
every day at 06:00 UTC automatically.

---

## Updating your interests

**If using Zotero export:** Re-export your library, replace the file in
`config/`, push. The next run picks it up automatically.

**If using text description:** Edit `interest_description` in `config.yaml`,
push.

---

## Tuning

| Setting | Effect |
|---|---|
| `threshold` | Cosine similarity cutoff (0–1). Raise to see fewer, more focused papers. |
| `top_n` | Always include this many top-scoring papers, even below threshold. |
| `max_age_days` | Ignore feed entries older than this. |
| `passthrough_feeds` | Feed names listed here are included unfiltered (good for low-volume journals). |

---

## Running locally

```bash
pip install -r requirements.txt
python scripts/filter.py --config config/config.yaml

# Preview without writing output:
python scripts/filter.py --dry-run
```

---

## How it works

1. Fetches each RSS feed and collects titles + abstracts.
2. Loads `all-MiniLM-L6-v2` (a small, fast sentence-transformer model) from
   the HuggingFace cache.
3. Builds an interest profile vector — either the mean embedding of your Zotero
   library, or the embedding of your text description.
4. Scores each paper by cosine similarity to the profile vector.
5. Keeps papers above `threshold`, always including the top `top_n`.
6. Writes a standard RSS XML file with relevance scores in the titles.
7. GitHub Pages serves the file; GitHub Actions re-runs daily.

The relevance score appears in brackets in each item title, e.g. `[73%]`, so
you can see at a glance how confident the filter is.
