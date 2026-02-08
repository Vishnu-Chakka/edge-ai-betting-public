"""
Stats data service — fetches player/team statistics from free APIs.
Supports NBA, NFL, MLB, NHL, and Soccer.
"""
from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class StatsService:
    """
    Aggregates statistics from multiple free data sources.
    Falls back to mock data if libraries aren't installed.
    """

    def __init__(self):
        self._nba_available = False
        self._nfl_available = False
        self._mlb_available = False
        self._hockey_available = False

        try:
            from nba_api.stats.endpoints import leaguegamefinder, teamgamelog
            self._nba_available = True
        except ImportError:
            logger.info("nba_api not installed — NBA stats will use mock data")

        try:
            import nfl_data_py
            self._nfl_available = True
        except ImportError:
            logger.info("nfl_data_py not installed — NFL stats will use mock data")

        try:
            import pybaseball
            self._mlb_available = True
        except ImportError:
            logger.info("pybaseball not installed — MLB stats will use mock data")

    async def get_team_stats(
        self, sport: str, team_id: str, num_games: int = 20
    ) -> Dict[str, Any]:
        """Get recent team statistics."""
        if sport == "nba":
            return await self._get_nba_team_stats(team_id, num_games)
        elif sport == "nfl":
            return await self._get_nfl_team_stats(team_id, num_games)
        elif sport == "mlb":
            return await self._get_mlb_team_stats(team_id, num_games)
        else:
            return self._mock_team_stats(sport, team_id, num_games)

    async def get_team_schedule(
        self, sport: str, team_id: str
    ) -> List[Dict[str, Any]]:
        """Get team's recent schedule for rest day calculations."""
        # For now return mock schedule
        return self._mock_schedule(sport, team_id)

    async def get_injuries(
        self, sport: str, league_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get current injury report."""
        # Injury data typically requires scraping — return mock for now
        return self._mock_injuries(sport)

    async def _get_nba_team_stats(
        self, team_id: str, num_games: int
    ) -> Dict[str, Any]:
        """Fetch NBA team stats from nba_api."""
        if not self._nba_available:
            return self._mock_team_stats("nba", team_id, num_games)

        try:
            from nba_api.stats.endpoints import teamgamelog
            from nba_api.stats.static import teams

            # nba_api is synchronous — run in executor in production
            log = teamgamelog.TeamGameLog(
                team_id=team_id, season="2025-26"
            )
            df = log.get_data_frames()[0]

            if len(df) == 0:
                return self._mock_team_stats("nba", team_id, num_games)

            recent = df.head(num_games)

            return {
                "team_id": team_id,
                "games_played": len(recent),
                "wins": int(recent["WL"].eq("W").sum()),
                "losses": int(recent["WL"].eq("L").sum()),
                "win_pct": float(recent["WL"].eq("W").mean()),
                "ppg": float(recent["PTS"].mean()),
                "opp_ppg": float(recent["PTS"].mean()) - float(recent["PLUS_MINUS"].mean()),
                "plus_minus": float(recent["PLUS_MINUS"].mean()),
                "fg_pct": float(recent["FG_PCT"].mean()),
                "fg3_pct": float(recent["FG3_PCT"].mean()),
                "ft_pct": float(recent["FT_PCT"].mean()),
                "reb": float(recent["REB"].mean()),
                "ast": float(recent["AST"].mean()),
                "tov": float(recent["TOV"].mean()),
                "stl": float(recent["STL"].mean()),
                "blk": float(recent["BLK"].mean()),
            }
        except Exception as e:
            logger.error(f"Error fetching NBA stats: {e}")
            return self._mock_team_stats("nba", team_id, num_games)

    async def _get_nfl_team_stats(
        self, team_id: str, num_games: int
    ) -> Dict[str, Any]:
        """Fetch NFL team stats from nfl_data_py."""
        if not self._nfl_available:
            return self._mock_team_stats("nfl", team_id, num_games)

        try:
            import nfl_data_py as nfl
            import pandas as pd

            # Get weekly data for current season
            weekly = nfl.import_weekly_data([2025])
            team_data = weekly[weekly["recent_team"] == team_id]

            if len(team_data) == 0:
                return self._mock_team_stats("nfl", team_id, num_games)

            recent = team_data.tail(num_games)

            return {
                "team_id": team_id,
                "games_played": len(recent),
                "passing_yards_pg": float(recent["passing_yards"].mean()),
                "rushing_yards_pg": float(recent["rushing_yards"].mean()),
                "passing_tds_pg": float(recent["passing_tds"].mean()),
                "rushing_tds_pg": float(recent["rushing_tds"].mean()),
                "interceptions_pg": float(recent["interceptions"].mean()),
                "fumbles_pg": float(recent.get("fumbles_lost", pd.Series([0])).mean()),
            }
        except Exception as e:
            logger.error(f"Error fetching NFL stats: {e}")
            return self._mock_team_stats("nfl", team_id, num_games)

    async def _get_mlb_team_stats(
        self, team_id: str, num_games: int
    ) -> Dict[str, Any]:
        """Fetch MLB team stats from pybaseball."""
        if not self._mlb_available:
            return self._mock_team_stats("mlb", team_id, num_games)

        try:
            from pybaseball import team_batting, team_pitching

            batting = team_batting(2025)
            pitching = team_pitching(2025)

            team_bat = batting[batting["Team"] == team_id]
            team_pitch = pitching[pitching["Team"] == team_id]

            if len(team_bat) == 0:
                return self._mock_team_stats("mlb", team_id, num_games)

            return {
                "team_id": team_id,
                "batting_avg": float(team_bat["AVG"].iloc[0]),
                "obp": float(team_bat["OBP"].iloc[0]),
                "slg": float(team_bat["SLG"].iloc[0]),
                "ops": float(team_bat["OPS"].iloc[0]),
                "runs_pg": float(team_bat["R"].iloc[0]) / max(float(team_bat["G"].iloc[0]), 1),
                "era": float(team_pitch["ERA"].iloc[0]) if len(team_pitch) > 0 else 4.00,
                "whip": float(team_pitch["WHIP"].iloc[0]) if len(team_pitch) > 0 else 1.30,
            }
        except Exception as e:
            logger.error(f"Error fetching MLB stats: {e}")
            return self._mock_team_stats("mlb", team_id, num_games)

    def _mock_team_stats(
        self, sport: str, team_id: str, num_games: int
    ) -> Dict[str, Any]:
        """Return mock stats for development."""
        base = {
            "team_id": team_id,
            "games_played": num_games,
            "data_source": "mock",
        }

        if sport == "nba":
            base.update({
                "wins": 12, "losses": 8, "win_pct": 0.600,
                "ppg": 112.5, "opp_ppg": 108.3, "plus_minus": 4.2,
                "off_rtg": 115.2, "def_rtg": 110.8, "pace": 100.3,
                "fg_pct": 0.475, "fg3_pct": 0.368, "ft_pct": 0.792,
                "reb": 44.5, "ast": 25.8, "tov": 13.2,
            })
        elif sport == "nfl":
            base.update({
                "wins": 8, "losses": 5, "win_pct": 0.615,
                "ppg": 24.6, "opp_ppg": 20.1,
                "passing_yards_pg": 245.3, "rushing_yards_pg": 118.7,
                "yards_pg": 364.0, "opp_yards_pg": 325.5,
                "turnover_diff": 0.4,
            })
        elif sport == "mlb":
            base.update({
                "wins": 55, "losses": 45, "win_pct": 0.550,
                "runs_pg": 4.8, "opp_runs_pg": 4.2,
                "batting_avg": 0.258, "obp": 0.332, "slg": 0.425,
                "era": 3.85, "whip": 1.22,
            })
        elif sport == "nhl":
            base.update({
                "wins": 28, "losses": 18, "otl": 5, "win_pct": 0.598,
                "goals_pg": 3.2, "goals_against_pg": 2.7,
                "pp_pct": 0.228, "pk_pct": 0.812,
                "shots_pg": 31.5, "shots_against_pg": 28.9,
            })
        else:  # soccer
            base.update({
                "wins": 12, "draws": 5, "losses": 3, "win_pct": 0.600,
                "goals_pg": 1.85, "goals_against_pg": 0.95,
                "xg_pg": 1.72, "xga_pg": 1.05,
                "possession_pct": 0.548,
                "shots_on_target_pg": 5.2,
            })

        return base

    def _mock_schedule(
        self, sport: str, team_id: str
    ) -> List[Dict[str, Any]]:
        """Mock schedule for rest day calculations."""
        from datetime import timedelta

        now = datetime.utcnow()
        games = []
        for i in range(10):
            games.append({
                "team_id": team_id,
                "date": (now - timedelta(days=i * 2 + 1)).isoformat(),
                "opponent": f"opponent_{i}",
                "home": i % 2 == 0,
                "result": "W" if i % 3 != 0 else "L",
                "score_for": 105 + i,
                "score_against": 100 + i,
            })
        return games

    def _mock_injuries(self, sport: str) -> List[Dict[str, Any]]:
        """Mock injury data."""
        return [
            {
                "player": "Star Player",
                "team": "Home Team",
                "status": "Questionable",
                "injury": "Ankle",
                "impact_score": 0.15,
            },
            {
                "player": "Key Defender",
                "team": "Away Team",
                "status": "OUT",
                "injury": "Knee",
                "impact_score": 0.25,
            },
        ]
