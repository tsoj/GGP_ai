# gavel_games

258 `.lud` game definitions evolved by the GAVEL project (Todd et al.,
NeurIPS 2024 — *Generating Games via Evolution and Language Models*),
which uses a fine-tuned LLM to mutate Ludii games and a UCT-based fitness
function to select the survivors.

These are the dedup-survivors of the main GAVEL experiment archive,
greedily filtered at cosine similarity ≥ 0.98 over the games' concept
embeddings (`gavel/inspect_archives/dedup_export.py`), then renamed
`game_<idx>_fit<fitness>_<run>.lud` and ordered by descending fitness.
`manifest.tsv` maps each filename back to its source archive entry.

This snapshot is committed here so the dataset generator does not depend
on a local `gavel/` checkout. Upstream:
- Project: <https://github.com/twni2016/gavel> (Todd et al. 2024)
- Original location in this repo: `gavel/selected_games_cos098/`
