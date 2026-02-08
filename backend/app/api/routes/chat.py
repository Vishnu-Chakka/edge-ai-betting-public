"""Chat API routes — conversation with Claude-powered EDGE AI."""
from __future__ import annotations

import json
import uuid
import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...services.chat_service import ChatService, build_context
from ...services.odds_service import OddsService
from ...config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory stores (swap for DB in production)
_conversations: Dict[str, List[Dict[str, str]]] = {}
_chat_service: Optional[ChatService] = None


def _get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


def get_odds_service() -> OddsService:
    """Dependency to create OddsService instance."""
    return OddsService(api_key=settings.ODDS_API_KEY)


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    stream: bool = False


class ChatResponse(BaseModel):
    conversation_id: str
    response: str


@router.post("/message")
async def send_message(
    req: ChatRequest,
    odds_service: OddsService = Depends(get_odds_service)
):
    """Send a message to EDGE AI and get a response."""
    svc = _get_chat_service()
    conv_id = req.conversation_id or str(uuid.uuid4())
    if conv_id not in _conversations:
        _conversations[conv_id] = []
    history = _conversations[conv_id]

    user_prefs = {
        "risk_tolerance": "moderate",
        "bankroll_amount": 1000,
        "bankroll_method": "fractional_kelly",
        "kelly_fraction": 0.25,
        "min_ev_threshold": 0.03,
        "enabled_sports": ["nba", "nfl", "mlb", "nhl", "soccer"],
    }

    # Fetch real picks data
    from .picks import _generate_picks
    picks_data = await _generate_picks(odds_service)
    await odds_service.close()

    # Convert picks format for chat context
    picks = []
    for pick in picks_data[:5]:  # Top 5 picks for chat context
        picks.append({
            "home_team": pick["game"]["home_team"],
            "away_team": pick["game"]["away_team"],
            "scheduled_at": pick["game"]["scheduled_at"],
            "display": pick["recommendation"]["display"],
            "best_odds": f"{pick['recommendation']['best_odds']:.2f}",
            "best_book": pick["recommendation"]["best_book"],
            "fair_prob": pick["recommendation"]["fair_prob"],
            "ev_pct": pick["recommendation"]["ev_pct"],
            "edge_type": pick["recommendation"]["edge_type"],
            "confidence_tier": pick["recommendation"]["confidence_tier"],
            "kelly_units": pick["recommendation"]["kelly_units"],
            "top_factors": pick["reasoning"]["top_factors"],
        })

    context = build_context(user_prefs, picks)

    if req.stream:
        return StreamingResponse(
            _stream(svc, conv_id, req.message, history, context),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Conversation-Id": conv_id},
        )

    response_text = svc.chat(
        user_message=req.message,
        conversation_history=history,
        context=context,
    )
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": response_text})
    if len(history) > 40:
        _conversations[conv_id] = history[-20:]
    return ChatResponse(conversation_id=conv_id, response=response_text)


async def _stream(svc, conv_id, message, history, context):
    full = []
    try:
        async for chunk in svc.chat_stream(
            user_message=message, conversation_history=history, context=context
        ):
            full.append(chunk)
            yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        fb = svc._fallback_response(message)
        yield f"data: {json.dumps({'type': 'text', 'content': fb})}\n\n"
        full.append(fb)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "".join(full)})
    yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id})}\n\n"


@router.get("/")
async def list_conversations():
    return {
        "conversations": [
            {"id": cid, "messages": len(msgs)}
            for cid, msgs in _conversations.items()
        ]
    }


@router.get("/{conversation_id}/messages")
async def get_messages(conversation_id: str):
    return {"messages": _conversations.get(conversation_id, [])}
