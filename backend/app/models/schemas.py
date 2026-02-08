from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.types import JSON


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.utcnow()


# ── Base ─────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


# ── User ─────────────────────────────────────────────────────────────────────


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    display_name: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )

    # relationships
    preferences: Mapped[Optional["UserPreference"]] = relationship(
        "UserPreference", back_populates="user", uselist=False, lazy="selectin"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="user", lazy="selectin"
    )
    bet_recommendations: Mapped[list["BetRecommendation"]] = relationship(
        "BetRecommendation", back_populates="user", lazy="selectin"
    )


# ── UserPreference ───────────────────────────────────────────────────────────


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), unique=True, nullable=False
    )

    # Sport & bet-type filters (stored as JSON arrays)
    enabled_sports: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None
    )
    bet_types: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None
    )

    # Risk / bankroll
    risk_tolerance: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, default="moderate"
    )
    bankroll_amount: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, default=None
    )
    bankroll_method: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, default="kelly"
    )
    kelly_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, default=0.25
    )
    max_bet_pct: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, default=0.05
    )
    min_ev_threshold: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, default=0.02
    )

    user: Mapped["User"] = relationship("User", back_populates="preferences")


# ── Sport ────────────────────────────────────────────────────────────────────


class Sport(Base):
    __tablename__ = "sports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    leagues: Mapped[list["League"]] = relationship(
        "League", back_populates="sport", lazy="selectin"
    )


# ── League ───────────────────────────────────────────────────────────────────


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sport_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sports.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    country: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    sport: Mapped["Sport"] = relationship("Sport", back_populates="leagues")
    teams: Mapped[list["Team"]] = relationship(
        "Team", back_populates="league", lazy="selectin"
    )
    games: Mapped[list["Game"]] = relationship(
        "Game", back_populates="league", lazy="selectin"
    )


# ── Team ─────────────────────────────────────────────────────────────────────


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    league_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("leagues.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    abbreviation: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True
    )
    elo_rating: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, default=1500.0
    )
    meta: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None
    )

    league: Mapped["League"] = relationship("League", back_populates="teams")


# ── Game ─────────────────────────────────────────────────────────────────────


class Game(Base):
    __tablename__ = "games"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    external_id: Mapped[Optional[str]] = mapped_column(
        String(128), unique=True, nullable=True, index=True
    )
    league_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("leagues.id"), nullable=False
    )
    home_team_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("teams.id"), nullable=False
    )
    away_team_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("teams.id"), nullable=False
    )
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32), default="scheduled", nullable=False
    )
    home_score: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    away_score: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    context: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None
    )

    league: Mapped["League"] = relationship("League", back_populates="games")
    home_team: Mapped["Team"] = relationship(
        "Team", foreign_keys=[home_team_id], lazy="selectin"
    )
    away_team: Mapped["Team"] = relationship(
        "Team", foreign_keys=[away_team_id], lazy="selectin"
    )
    odds_snapshots: Mapped[list["OddsSnapshot"]] = relationship(
        "OddsSnapshot", back_populates="game", lazy="selectin"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="game", lazy="selectin"
    )
    bet_recommendations: Mapped[list["BetRecommendation"]] = relationship(
        "BetRecommendation", back_populates="game", lazy="selectin"
    )


# ── OddsSnapshot ─────────────────────────────────────────────────────────────


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    game_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("games.id"), nullable=False, index=True
    )
    sportsbook: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    market_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )

    # Moneyline odds
    home_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    draw_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Spread / total / props as JSON blobs
    spreads: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None
    )
    totals: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None
    )
    props: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )
    is_closing_line: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    game: Mapped["Game"] = relationship(
        "Game", back_populates="odds_snapshots"
    )


# ── Prediction ───────────────────────────────────────────────────────────────


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    game_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("games.id"), nullable=False, index=True
    )
    market_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    model_name: Mapped[str] = mapped_column(
        String(64), nullable=False
    )

    home_win_prob: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    away_win_prob: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    draw_prob: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    predicted_spread: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    predicted_total: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    top_features: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None
    )
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )

    game: Mapped["Game"] = relationship("Game", back_populates="predictions")


# ── BetRecommendation ────────────────────────────────────────────────────────


class BetRecommendation(Base):
    __tablename__ = "bet_recommendations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    game_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("games.id"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )

    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    bet_side: Mapped[str] = mapped_column(String(32), nullable=False)

    # Book & odds
    recommended_book: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    recommended_odds: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    # Probabilities
    model_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    implied_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Edge metrics
    ev_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    kelly_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    recommended_units: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    recommended_amount: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    edge_type: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )
    confidence_tier: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True
    )

    # Lifecycle
    status: Mapped[str] = mapped_column(
        String(32), default="pending", nullable=False
    )

    # CLV (closing-line value) tracking
    opening_odds: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    closing_odds: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    clv_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # P&L
    actual_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )

    game: Mapped["Game"] = relationship(
        "Game", back_populates="bet_recommendations"
    )
    user: Mapped["User"] = relationship(
        "User", back_populates="bet_recommendations"
    )


# ── Conversation ─────────────────────────────────────────────────────────────


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )
    meta: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None
    )

    user: Mapped["User"] = relationship(
        "User", back_populates="conversations"
    )
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", lazy="selectin"
    )


# ── Message ──────────────────────────────────────────────────────────────────


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(
        String(16), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Token accounting
    input_tokens: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    output_tokens: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    meta: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )


# ── ModelPerformance ─────────────────────────────────────────────────────────


class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    league_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("leagues.id"), nullable=True
    )
    market_type: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )

    # Evaluation window
    eval_start: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    eval_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )

    # Calibration & discrimination metrics
    brier_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    log_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    auc_roc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Betting-specific metrics
    roi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clv_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_size: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )

    league: Mapped[Optional["League"]] = relationship("League")
