"""User preferences API routes."""
from __future__ import annotations
from typing import Optional, Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# In-memory user prefs (swap for DB in production)
_user_prefs: Dict[str, Dict[str, Any]] = {
    "default": {
        "enabled_sports": ["nba", "nfl", "mlb", "nhl", "soccer"],
        "bet_types": ["moneyline", "spread", "totals"],
        "risk_tolerance": "moderate",
        "bankroll_amount": 1000.00,
        "bankroll_method": "fractional_kelly",
        "kelly_fraction": 0.25,
        "max_bet_pct": 0.05,
        "min_ev_threshold": 0.03,
    }
}

class UserPrefsUpdate(BaseModel):
    enabled_sports: Optional[list] = None
    bet_types: Optional[list] = None
    risk_tolerance: Optional[str] = None
    bankroll_amount: Optional[float] = None
    bankroll_method: Optional[str] = None
    kelly_fraction: Optional[float] = None
    max_bet_pct: Optional[float] = None
    min_ev_threshold: Optional[float] = None

@router.get("/preferences")
async def get_preferences():
    return {"preferences": _user_prefs.get("default", {})}

@router.put("/preferences")
async def update_preferences(update: UserPrefsUpdate):
    current = _user_prefs.get("default", {})
    for field, value in update.dict(exclude_none=True).items():
        current[field] = value
    _user_prefs["default"] = current
    return {"preferences": current, "status": "updated"}

@router.get("/performance")
async def get_user_performance():
    return {
        "performance": {
            "total_bets": 47, "wins": 26, "losses": 21,
            "win_rate": 0.553, "roi": 0.068,
            "total_wagered": 2350.00, "total_pnl": 159.80,
            "avg_ev": 0.045, "avg_clv": 0.019,
            "current_bankroll": 1159.80,
            "best_sport": "nba",
            "best_tier": "A",
        }
    }

@router.get("/me")
async def get_current_user():
    return {
        "user": {
            "id": "demo_user",
            "email": "demo@edge-ai.com",
            "display_name": "Demo User",
            "preferences": _user_prefs.get("default", {}),
        }
    }
