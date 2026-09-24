"""Download StatsBomb open data (https://github.com/statsbomb/open-data).

Only two kinds of file are needed:
  matches/<competition>/<season>.json   fixtures, scores and the manager of each side
  events/<match_id>.json                every on-ball event of a match (~3 MB each)

Event files are large, so the dataset builder turns each one into two rows of
features straight away and never keeps the raw events around.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# 2015/16 is the one season StatsBomb released in full for the big leagues.
# (Its Bundesliga 2015/16 file only has Leverkusen's matches, so it is left out.)
LEAGUES_2015_16 = {
    "Premier League": (2, 27),
    "La Liga": (11, 27),
    "Serie A": (12, 27),
    "Ligue 1": (7, 27),
}


def get_json(url: str, retries: int = 4):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def matches(competition_id: int, season_id: int, cache_dir: Path | None = None) -> list[dict]:
    path = cache_dir / f"matches_{competition_id}_{season_id}.json" if cache_dir else None
    if path and path.exists():
        return json.loads(path.read_text())
    data = get_json(f"{BASE}/matches/{competition_id}/{season_id}.json")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return data


def events(match_id: int) -> list[dict]:
    return get_json(f"{BASE}/events/{match_id}.json")


def manager_name(side: dict) -> str | None:
    """The manager listed for one side of a match ("A / B" if two are listed)."""
    names = [m.get("nickname") or m["name"] for m in side.get("managers") or []]
    return " / ".join(names) if names else None
