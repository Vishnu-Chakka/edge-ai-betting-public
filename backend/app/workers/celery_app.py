"""
Celery application configuration for background task workers.
Handles odds ingestion, stats updates, and model retraining on schedule.
"""
from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

# Celery configuration
# In production, use Redis as broker. For dev, tasks run synchronously.
CELERY_BROKER_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

try:
    from celery import Celery
    from celery.schedules import crontab

    app = Celery(
        "edge_ai",
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND,
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=300,  # 5 min max per task
        worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (memory leak prevention)
    )

    # Beat schedule for periodic tasks
    app.conf.beat_schedule = {
        # Odds ingestion: 4x per day (every 6 hours)
        "ingest-nba-odds": {
            "task": "app.workers.odds_worker.ingest_sport_odds",
            "schedule": crontab(minute=0, hour="*/6"),
            "args": ("nba",),
        },
        "ingest-nfl-odds": {
            "task": "app.workers.odds_worker.ingest_sport_odds",
            "schedule": crontab(minute=15, hour="*/6"),
            "args": ("nfl",),
        },
        "ingest-mlb-odds": {
            "task": "app.workers.odds_worker.ingest_sport_odds",
            "schedule": crontab(minute=30, hour="*/6"),
            "args": ("mlb",),
        },
        "ingest-nhl-odds": {
            "task": "app.workers.odds_worker.ingest_sport_odds",
            "schedule": crontab(minute=45, hour="*/6"),
            "args": ("nhl",),
        },
        "ingest-soccer-odds": {
            "task": "app.workers.odds_worker.ingest_sport_odds",
            "schedule": crontab(minute=0, hour="3,9,15,21"),
            "args": ("epl",),
        },
        # Stats update: daily at 4 AM UTC
        "update-daily-stats": {
            "task": "app.workers.stats_worker.update_all_stats",
            "schedule": crontab(minute=0, hour=4),
        },
        # Elo ratings update: daily at 5 AM UTC (after stats)
        "update-elo-ratings": {
            "task": "app.workers.stats_worker.update_elo_ratings",
            "schedule": crontab(minute=0, hour=5),
        },
    }

    CELERY_AVAILABLE = True

except ImportError:
    logger.info("Celery not installed — background tasks disabled. Tasks will run inline.")
    CELERY_AVAILABLE = False
    app = None
