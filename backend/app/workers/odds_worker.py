"""
Odds ingestion worker tasks.
Fetches odds from The Odds API and stores in database.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import Celery task decorator; fall back to plain function
try:
    from .celery_app import app as celery_app
    if celery_app:
        task = celery_app.task
    else:
        def task(**kwargs):
            def decorator(func):
                return func
            return decorator
except ImportError:
    def task(**kwargs):
        def decorator(func):
            return func
        return decorator


@task(name="app.workers.odds_worker.ingest_sport_odds", bind=True, max_retries=2)
def ingest_sport_odds(self, sport: str, markets: str = "h2h"):
    """
    Celery task: Fetch and store odds for a given sport.

    Args:
        sport: Internal sport key (nba, nfl, mlb, nhl, epl, etc.)
        markets: Comma-separated markets to fetch (h2h, spreads, totals)
    """
    logger.info(f"Starting odds ingestion for {sport} (markets={markets})")

    try:
        # Run the async ingestion in an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_async_ingest(sport, markets))
        loop.close()

        logger.info(
            f"Odds ingestion complete for {sport}: "
            f"{result.get('games_count', 0)} games, "
            f"{result.get('snapshots_count', 0)} snapshots"
        )
        return result

    except Exception as e:
        logger.error(f"Odds ingestion failed for {sport}: {e}")
        if hasattr(self, 'retry'):
            raise self.retry(countdown=60, exc=e)
        raise


async def _async_ingest(sport: str, markets: str) -> dict:
    """Async implementation of odds ingestion."""
    from ..services.odds_service import OddsService
    from ..pipeline.normalizer import OddsNormalizer

    odds_service = OddsService()
    normalizer = OddsNormalizer()

    try:
        # Fetch raw odds
        raw_games = await odds_service.fetch_sport_odds(sport, markets=markets)

        if not raw_games:
            logger.warning(f"No odds data returned for {sport}")
            return {"games_count": 0, "snapshots_count": 0}

        # Parse into snapshots
        snapshots = odds_service.parse_odds_response(raw_games)

        # Find best odds per game
        game_ids = set(s["game_external_id"] for s in snapshots)
        best_odds = {}
        for gid in game_ids:
            game_snaps = [s for s in snapshots if s["game_external_id"] == gid]
            best_odds[gid] = odds_service.compute_best_odds(game_snaps)

            # Compute no-vig consensus
            h2h_snaps = [
                s for s in game_snaps
                if s.get("home_odds") and s.get("away_odds")
            ]
            if h2h_snaps:
                home_odds = [s["home_odds"] for s in h2h_snaps]
                away_odds = [s["away_odds"] for s in h2h_snaps]
                books = [s["sportsbook"] for s in h2h_snaps]

                consensus = normalizer.compute_consensus_line_from_lists(
                    home_odds, away_odds, books
                )
                best_odds[gid]["consensus"] = consensus

        logger.info(
            f"Processed {len(game_ids)} games with "
            f"{len(snapshots)} total snapshots for {sport}"
        )

        # TODO: Store snapshots in database when DB is connected
        # For now, log the results
        for gid, odds in best_odds.items():
            logger.debug(
                f"  Game {gid}: Best home={odds['home']}, "
                f"Best away={odds['away']}"
            )

        return {
            "games_count": len(game_ids),
            "snapshots_count": len(snapshots),
            "best_odds": {
                gid: {
                    "home_odds": odds["home"]["odds"],
                    "home_book": odds["home"]["book"],
                    "away_odds": odds["away"]["odds"],
                    "away_book": odds["away"]["book"],
                }
                for gid, odds in best_odds.items()
            },
        }

    finally:
        await odds_service.close()


@task(name="app.workers.odds_worker.ingest_all_odds")
def ingest_all_odds():
    """Ingest odds for all configured sports."""
    sports = ["nba", "nfl", "mlb", "nhl", "epl"]
    results = {}
    for sport in sports:
        try:
            result = ingest_sport_odds(sport)
            results[sport] = result
        except Exception as e:
            logger.error(f"Failed to ingest {sport}: {e}")
            results[sport] = {"error": str(e)}
    return results
