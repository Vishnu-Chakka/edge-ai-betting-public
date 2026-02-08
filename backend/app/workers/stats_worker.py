"""
Stats ingestion worker tasks.
Updates team statistics, Elo ratings, and model predictions.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

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


@task(name="app.workers.stats_worker.update_all_stats")
def update_all_stats():
    """Update statistics for all sports."""
    logger.info("Starting daily stats update for all sports")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(_async_update_stats())
    loop.close()

    logger.info(f"Stats update complete: {result}")
    return result


async def _async_update_stats() -> dict:
    """Async implementation of stats update."""
    from ..services.stats_service import StatsService

    stats_service = StatsService()
    results = {}

    sports = ["nba", "nfl", "mlb", "nhl", "soccer"]
    for sport in sports:
        try:
            # Fetch team stats (mock data for now)
            stats = await stats_service.get_team_stats(
                sport, f"mock_team_{sport}", num_games=20
            )
            results[sport] = {
                "status": "ok",
                "games_found": stats.get("games_played", 0),
            }
        except Exception as e:
            logger.error(f"Stats update failed for {sport}: {e}")
            results[sport] = {"status": "error", "message": str(e)}

    return results


@task(name="app.workers.stats_worker.update_elo_ratings")
def update_elo_ratings():
    """Update Elo ratings based on recent results."""
    logger.info("Starting Elo ratings update")

    from ..ml.elo import EloRatingSystem

    sports = ["nba", "nfl", "mlb", "nhl", "soccer"]
    results = {}

    for sport in sports:
        try:
            elo = EloRatingSystem(sport)
            # Try to load existing ratings
            elo.load_ratings(f"data/elo_{sport}.json")

            # TODO: Fetch recent game results and update ratings
            # For now just report current state
            results[sport] = {
                "status": "ok",
                "teams_rated": len(elo.ratings),
            }

            # Save updated ratings
            elo.save_ratings(f"data/elo_{sport}.json")

        except Exception as e:
            logger.error(f"Elo update failed for {sport}: {e}")
            results[sport] = {"status": "error", "message": str(e)}

    logger.info(f"Elo update complete: {results}")
    return results


@task(name="app.workers.stats_worker.refresh_predictions")
def refresh_predictions():
    """Refresh model predictions for all upcoming games."""
    logger.info("Refreshing predictions for upcoming games")

    # This would:
    # 1. Load all games scheduled for today/tomorrow
    # 2. Compute features for each game
    # 3. Run all models (Elo, Poisson, ML, Bayesian)
    # 4. Combine with ensemble
    # 5. Compare against current odds to find EV
    # 6. Store predictions and recommendations

    return {"status": "ok", "note": "Full prediction pipeline not yet connected to DB"}
