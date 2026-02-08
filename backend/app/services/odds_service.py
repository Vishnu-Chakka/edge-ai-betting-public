"""
The Odds API client service.
Handles fetching odds, budget management, and caching.
"""
from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import httpx

logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"

SPORT_KEYS = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "epl": "soccer_epl",
    "la_liga": "soccer_spain_la_liga",
    "bundesliga": "soccer_germany_bundesliga",
    "serie_a": "soccer_italy_serie_a",
    "ligue_1": "soccer_france_ligue_one",
    "mls": "soccer_usa_mls",
}

SHARPNESS_WEIGHTS = {
    "pinnacle": 1.0,
    "betfair_ex_eu": 0.95,
    "matchbook": 0.85,
    "bookmaker": 0.80,
    "bovada": 0.60,
    "draftkings": 0.55,
    "fanduel": 0.50,
    "betmgm": 0.45,
    "pointsbetus": 0.40,
    "williamhill_us": 0.50,
    "unibet_us": 0.45,
    "betrivers": 0.40,
    "superbook": 0.45,
    "lowvig": 0.80,
    "betonlineag": 0.55,
}


class OddsService:
    """Client for The Odds API with budget tracking."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        self.remaining_requests: Optional[int] = None
        self._client = httpx.AsyncClient(timeout=30.0)

    async def fetch_odds(
        self,
        sport_key: str,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "decimal",
    ) -> List[Dict[str, Any]]:
        """
        Fetch odds from The Odds API for a given sport.

        Args:
            sport_key: The Odds API sport key (e.g., 'basketball_nba')
            regions: Comma-separated regions (us, uk, eu, au)
            markets: Comma-separated markets (h2h, spreads, totals)
            odds_format: 'decimal' or 'american'

        Returns:
            List of game dicts with bookmaker odds
        """
        if not self.api_key:
            logger.warning("No ODDS_API_KEY set — returning mock data")
            return self._mock_odds(sport_key)

        if self.remaining_requests is not None and self.remaining_requests < 10:
            logger.warning(
                f"Odds API budget critically low: {self.remaining_requests} remaining"
            )
            return []

        try:
            response = await self._client.get(
                f"{ODDS_API_BASE}/{sport_key}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": odds_format,
                },
            )
            response.raise_for_status()

            # Track remaining requests from headers
            remaining = response.headers.get("x-requests-remaining")
            if remaining is not None:
                self.remaining_requests = int(remaining)
                logger.info(f"Odds API requests remaining: {self.remaining_requests}")

            used = response.headers.get("x-requests-used")
            if used:
                logger.info(f"Odds API requests used: {used}")

            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Odds API HTTP error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Odds API error: {e}")
            return []

    async def fetch_sport_odds(
        self, sport: str, markets: str = "h2h"
    ) -> List[Dict[str, Any]]:
        """Fetch odds using our internal sport key mapping."""
        sport_key = SPORT_KEYS.get(sport)
        if not sport_key:
            logger.warning(f"Unknown sport: {sport}")
            return []
        return await self.fetch_odds(sport_key, markets=markets)

    async def fetch_all_sports_odds(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch odds for all configured sports. Budget-aware."""
        results = {}
        for sport, sport_key in SPORT_KEYS.items():
            if self.remaining_requests is not None and self.remaining_requests < 20:
                logger.warning("Stopping odds fetch — budget low")
                break
            odds = await self.fetch_odds(sport_key)
            if odds:
                results[sport] = odds
        return results

    def parse_odds_response(
        self, games: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse raw Odds API response into normalized snapshots.

        Returns list of dicts with:
            game_external_id, home_team, away_team, scheduled_at,
            sportsbook, market_type, home_odds, away_odds, draw_odds
        """
        snapshots = []
        now = datetime.utcnow().isoformat()

        for game in games:
            game_id = game.get("id", "")
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            commence_time = game.get("commence_time", "")

            for bookmaker in game.get("bookmakers", []):
                book_key = bookmaker.get("key", "")
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")
                    outcomes = {
                        o["name"]: o.get("price")
                        for o in market.get("outcomes", [])
                    }

                    snapshot = {
                        "game_external_id": game_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "scheduled_at": commence_time,
                        "sportsbook": book_key,
                        "market_type": market_key,
                        "home_odds": outcomes.get(home_team),
                        "away_odds": outcomes.get(away_team),
                        "draw_odds": outcomes.get("Draw"),
                        "recorded_at": now,
                    }

                    # Add spread/total fields if present
                    if market_key == "spreads":
                        for o in market.get("outcomes", []):
                            if o["name"] == home_team:
                                snapshot["home_spread"] = o.get("point")
                                snapshot["home_spread_odds"] = o.get("price")
                            elif o["name"] == away_team:
                                snapshot["away_spread"] = o.get("point")
                                snapshot["away_spread_odds"] = o.get("price")

                    if market_key == "totals":
                        for o in market.get("outcomes", []):
                            if o["name"] == "Over":
                                snapshot["total_line"] = o.get("point")
                                snapshot["over_odds"] = o.get("price")
                            elif o["name"] == "Under":
                                snapshot["under_odds"] = o.get("price")

                    snapshots.append(snapshot)

        return snapshots

    def compute_best_odds(
        self, snapshots: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Find best available odds across all books for each side.

        Returns dict keyed by side ('home', 'away', 'draw') with
        best odds and book name.
        """
        best = {
            "home": {"odds": 0.0, "book": ""},
            "away": {"odds": 0.0, "book": ""},
            "draw": {"odds": 0.0, "book": ""},
        }

        for snap in snapshots:
            if snap.get("home_odds") and snap["home_odds"] > best["home"]["odds"]:
                best["home"] = {"odds": snap["home_odds"], "book": snap["sportsbook"]}
            if snap.get("away_odds") and snap["away_odds"] > best["away"]["odds"]:
                best["away"] = {"odds": snap["away_odds"], "book": snap["sportsbook"]}
            if snap.get("draw_odds") and snap["draw_odds"] > best["draw"]["odds"]:
                best["draw"] = {"odds": snap["draw_odds"], "book": snap["sportsbook"]}

        return best

    def _mock_odds(self, sport_key: str) -> List[Dict[str, Any]]:
        """Return mock odds data for development without an API key."""
        mock_games = {
            "basketball_nba": [
                {
                    "id": "mock_nba_1",
                    "sport_key": sport_key,
                    "home_team": "Boston Celtics",
                    "away_team": "New York Knicks",
                    "commence_time": datetime.utcnow().isoformat() + "Z",
                    "bookmakers": [
                        {
                            "key": "fanduel",
                            "title": "FanDuel",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Boston Celtics", "price": 1.69},
                                        {"name": "New York Knicks", "price": 2.20},
                                    ],
                                }
                            ],
                        },
                        {
                            "key": "draftkings",
                            "title": "DraftKings",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Boston Celtics", "price": 1.67},
                                        {"name": "New York Knicks", "price": 2.25},
                                    ],
                                }
                            ],
                        },
                        {
                            "key": "pinnacle",
                            "title": "Pinnacle",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Boston Celtics", "price": 1.70},
                                        {"name": "New York Knicks", "price": 2.22},
                                    ],
                                }
                            ],
                        },
                    ],
                },
                {
                    "id": "mock_nba_2",
                    "sport_key": sport_key,
                    "home_team": "Los Angeles Lakers",
                    "away_team": "Golden State Warriors",
                    "commence_time": datetime.utcnow().isoformat() + "Z",
                    "bookmakers": [
                        {
                            "key": "fanduel",
                            "title": "FanDuel",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Los Angeles Lakers", "price": 1.91},
                                        {"name": "Golden State Warriors", "price": 1.91},
                                    ],
                                }
                            ],
                        },
                        {
                            "key": "draftkings",
                            "title": "DraftKings",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Los Angeles Lakers", "price": 1.87},
                                        {"name": "Golden State Warriors", "price": 1.95},
                                    ],
                                }
                            ],
                        },
                    ],
                },
            ],
        }
        return mock_games.get(sport_key, mock_games["basketball_nba"])

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
