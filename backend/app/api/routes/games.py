"""Games API routes."""
from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Query, Depends

from ...services.odds_service import OddsService
from ...config import settings

router = APIRouter()

def get_odds_service() -> OddsService:
    """Dependency to create OddsService instance."""
    return OddsService(api_key=settings.ODDS_API_KEY)

@router.get("/today")
async def list_todays_games(
    sport: str = Query("all"),
    odds_service: OddsService = Depends(get_odds_service)
):
    """Fetch today's games from the Odds API."""
    all_games = []

    if sport == "all":
        # Fetch for all sports
        sports_to_fetch = ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl"]
    else:
        # Map sport to API key
        sport_map = {
            "nba": "basketball_nba",
            "nfl": "americanfootball_nfl",
            "mlb": "baseball_mlb",
            "nhl": "icehockey_nhl"
        }
        sports_to_fetch = [sport_map.get(sport, "basketball_nba")]

    for sport_key in sports_to_fetch:
        odds_data = await odds_service.fetch_odds(sport_key)
        for game in odds_data:
            all_games.append({
                "id": game.get("id"),
                "sport": sport_key.split("_")[0],
                "league": sport_key.split("_")[1].upper(),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "scheduled_at": game.get("commence_time"),
                "status": "scheduled",
                "bookmakers": game.get("bookmakers", [])
            })

    await odds_service.close()
    return {"games": all_games, "count": len(all_games)}

@router.get("/{game_id}")
async def get_game(
    game_id: str,
    odds_service: OddsService = Depends(get_odds_service)
):
    """Get specific game details."""
    # Fetch from multiple sports to find the game
    for sport_key in ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl"]:
        odds_data = await odds_service.fetch_odds(sport_key)
        for game in odds_data:
            if game.get("id") == game_id:
                await odds_service.close()
                return {
                    "game": {
                        "id": game.get("id"),
                        "sport": sport_key.split("_")[0],
                        "league": sport_key.split("_")[1].upper(),
                        "home_team": game.get("home_team"),
                        "away_team": game.get("away_team"),
                        "scheduled_at": game.get("commence_time"),
                        "status": "scheduled",
                        "bookmakers": game.get("bookmakers", [])
                    }
                }

    await odds_service.close()
    return {"error": "Game not found"}

@router.get("/{game_id}/odds")
async def get_game_odds(
    game_id: str,
    odds_service: OddsService = Depends(get_odds_service)
):
    """Get best odds for a specific game."""
    # Fetch from multiple sports to find the game
    for sport_key in ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl"]:
        odds_data = await odds_service.fetch_odds(sport_key, markets="h2h")
        for game in odds_data:
            if game.get("id") == game_id:
                # Parse odds and find best odds
                snapshots = odds_service.parse_odds_response([game])
                best_odds = odds_service.compute_best_odds(snapshots)

                await odds_service.close()
                return {
                    "game_id": game_id,
                    "best_odds": {
                        "home": best_odds["home"],
                        "away": best_odds["away"],
                    },
                    "all_bookmakers": game.get("bookmakers", [])
                }

    await odds_service.close()
    return {"error": "Game not found"}

@router.get("/{game_id}/prediction")
async def get_game_prediction(game_id: str):
    """Get model prediction for a game. This uses ML models (not yet fully implemented)."""
    return {
        "game_id": game_id,
        "prediction": {
            "home_win_prob": 0.628,
            "away_win_prob": 0.372,
            "model_breakdown": {
                "elo": 0.615, "ml_ensemble": 0.641,
                "bayesian": 0.622, "market": 0.605,
            },
        },
        "note": "Predictions will be calculated using ML models once trained"
    }
