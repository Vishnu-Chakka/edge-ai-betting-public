# EDGE AI: Sports Betting Chatbot System Blueprint

> A professional-grade, +EV sports betting analysis engine powered by multi-layer statistical models, ML ensembles, and Claude AI.

**Target:** All major sports (NBA, NFL, MLB, NHL, Soccer)
**Deployment:** Cloud web app (Vercel + Railway)
**Budget:** Free-tier APIs + ~$15-30/month operating cost
**LLM:** Claude API (Anthropic) for conversational reasoning

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Tech Stack](#2-tech-stack)
3. [Database Schema](#3-database-schema)
4. [Data Pipeline](#4-data-pipeline)
5. [Modeling Architecture](#5-modeling-architecture)
6. [Bet Selection Engine](#6-bet-selection-engine)
7. [API Design](#7-api-design)
8. [Chatbot Prompt Engineering](#8-chatbot-prompt-engineering)
9. [Example Responses](#9-example-chatbot-responses)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Development Roadmap](#11-development-roadmap)
12. [Appendix: Key Trade-offs](#12-appendix)

---

## 1. System Architecture

### High-Level Component Diagram

```
+========================================================================+
|                         CLIENT LAYER                                    |
|  +-------------------+   +--------------------+   +-----------------+  |
|  | Next.js Frontend  |   | Mobile PWA         |   | WebSocket       |  |
|  | (React + TS)      |   | (Same codebase)    |   | Real-time Feed  |  |
|  +--------+----------+   +--------+-----------+   +--------+--------+  |
+===========|=======================|=========================|===========+
            v                       v                         v
+========================================================================+
|                       API GATEWAY LAYER                                 |
|  +------------------------------------------------------------------+  |
|  | Next.js API Routes / FastAPI Reverse Proxy                       |  |
|  | - Rate Limiting (per-user + global)                              |  |
|  | - JWT Auth (Clerk / NextAuth)                                    |  |
|  | - Request Validation (Zod / Pydantic)                            |  |
|  | - WebSocket Upgrade Handler                                      |  |
|  +------------------------------------------------------------------+  |
+========================================================================+
            |                   |                   |
            v                   v                   v
+========================================================================+
|                    SERVICE LAYER (Python Backend)                       |
|                                                                        |
|  +------------------+  +------------------+  +--------------------+    |
|  | CHATBOT SERVICE  |  | BETTING ENGINE   |  | DATA INGESTION     |   |
|  | (Claude API)     |  | SERVICE          |  | SERVICE             |   |
|  |                  |  |                  |  |                     |   |
|  | - Conversation   |  | - EV Calculator  |  | - Odds Fetcher     |   |
|  |   Manager        |  | - Kelly Sizer    |  | - Stats Scrapers   |   |
|  | - Context        |  | - Bet Ranker     |  | - Injury Tracker   |   |
|  |   Builder        |  | - CLV Tracker    |  | - Weather API      |   |
|  | - Prompt         |  | - Correlation    |  | - Line Movement    |   |
|  |   Templates      |  |   Detector       |  |   Recorder         |   |
|  +--------+---------+  +--------+---------+  +--------+-----------+   |
|           |                      |                      |              |
|           v                      v                      v              |
|  +------------------------------------------------------------------+  |
|  |                    MODEL INFERENCE LAYER                          |  |
|  |  +---------------+  +---------------+  +---------------------+   |  |
|  |  | Baseline      |  | ML Ensemble   |  | Bayesian Updater    |   |  |
|  |  | (Regression,  |  | (XGBoost, RF, |  | (Beta posteriors,   |   |  |
|  |  |  Poisson, Elo)|  |  LightGBM)    |  |  sequential update) |   |  |
|  |  +-------+-------+  +-------+-------+  +----------+----------+   |  |
|  |          +-------------------+----------------------+             |  |
|  |                    +---------v----------+                         |  |
|  |                    | ENSEMBLE COMBINER  |                         |  |
|  |                    | (Stacked meta-     |                         |  |
|  |                    |  learner / weighted|                         |  |
|  |                    |  average)          |                         |  |
|  |                    +--------------------+                         |  |
|  +------------------------------------------------------------------+  |
+========================================================================+
            |                                  |
            v                                  v
+========================================================================+
|                        DATA LAYER                                       |
|  +------------------+  +------------------+  +--------------------+    |
|  | PostgreSQL       |  | Redis            |  | S3 / MinIO         |   |
|  | (Supabase)       |  | (Upstash)        |  | (Cloudflare R2)    |   |
|  |                  |  |                  |  |                     |   |
|  | - Game data      |  | - Odds cache     |  | - Trained models   |   |
|  | - Player stats   |  | - Session state  |  | - Feature stores   |   |
|  | - Bet history    |  | - Rate limits    |  | - Backtest results |   |
|  | - User prefs     |  | - Pub/Sub for WS |  | - Historical odds  |   |
|  | - Odds snapshots |  |                  |  |   snapshots         |   |
|  +------------------+  +------------------+  +--------------------+    |
+========================================================================+
```

### Request Flow Example

**"Give me your best NBA play tonight":**

1. User message arrives at Next.js API route
2. Route authenticates via JWT, extracts user preferences
3. Request forwarded to FastAPI Chatbot Service
4. Context Builder queries Redis (cached odds), PostgreSQL (user prefs, history), Betting Engine (ranked plays)
5. Betting Engine loads pre-computed predictions from Model Inference Layer
6. Context assembled with structured XML tags, sent to Claude API
7. Claude generates natural-language response with bet recommendation
8. Response streamed back via SSE (Server-Sent Events)

---

## 2. Tech Stack

### Frontend
| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Framework | **Next.js 15 (App Router)** | SSR, API routes, streaming support |
| Language | **TypeScript** | Type safety across API contracts |
| Styling | **Tailwind CSS + shadcn/ui** | Rapid UI, accessible components |
| State | **Zustand** | Lightweight for chat state |
| Charts | **Recharts or Nivo** | Odds movement, EV distributions |
| Real-time | **Socket.io client** | Live odds update streaming |
| Auth | **Clerk or NextAuth.js** | Zero-config JWT auth |

### Backend
| Component | Technology | Rationale |
|-----------|-----------|-----------|
| API Framework | **FastAPI (Python 3.12+)** | Async, Pydantic validation, auto-docs |
| Task Queue | **Celery + Redis** | Scheduled ingestion, retraining jobs |
| Scheduler | **APScheduler / Celery Beat** | Cron-like odds polling & model updates |
| WebSocket | **FastAPI WebSocket + Redis Pub/Sub** | Real-time alerts |
| LLM Client | **anthropic Python SDK** | Official Claude API client |

### ML / Data Science
| Component | Technology | Rationale |
|-----------|-----------|-----------|
| ML Framework | **scikit-learn, XGBoost, LightGBM** | Battle-tested for tabular data |
| Bayesian | **PyMC or scipy.stats (Beta)** | Conjugate prior updating |
| Data | **Pandas + Polars** | Polars for performance, Pandas for compat |
| Feature Store | **Custom (PostgreSQL + Redis)** | Free, no vendor lock-in |
| Serialization | **joblib / ONNX** | Fast inference load times |

### Data Sources (Free Tier)
| Sport | Library / API | Data | Rate Limits |
|-------|-------------|------|-------------|
| Odds (all) | **The Odds API** | Live odds, 40+ books | 500 req/month |
| NBA | **nba_api** | Player stats, game logs | Unofficial, courtesy limits |
| NFL | **nfl_data_py** | Play-by-play, weekly stats | GitHub-hosted, unlimited |
| MLB | **pybaseball** | Statcast, FanGraphs, BBRef | Scraping, courtesy limits |
| NHL | **hockey_scraper + nhl-api-py** | Play-by-play, shift data | Unofficial NHL API |
| Soccer | **football-data.org** | 12 leagues, fixtures | 10 req/min |
| Injuries | **Rotowire RSS / scrape** | Cross-sport injury reports | Scraping with cache |
| Weather | **Open-Meteo API** | Historical + forecast | Free, no key |

### Infrastructure
| Component | Technology | Free Tier Limits |
|-----------|-----------|-----------------|
| Frontend Hosting | **Vercel** | 100GB bandwidth/month |
| Backend Hosting | **Railway / Render / Fly.io** | 512MB RAM, shared CPU |
| Database | **Supabase PostgreSQL** | 500MB storage |
| Cache | **Upstash Redis** | 10K commands/day |
| Object Storage | **Cloudflare R2** | 10GB, 1M reads/month |
| CI/CD | **GitHub Actions** | Free for public repos |
| Monitoring | **Sentry + Grafana Cloud** | Free tiers |

---

## 3. Database Schema

### Core PostgreSQL Schema

```sql
-- ============================================================
-- USERS & PREFERENCES
-- ============================================================

CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    display_name    VARCHAR(100),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE user_preferences (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
    enabled_sports  JSONB DEFAULT '["nba","nfl","mlb","nhl","soccer"]',
    bet_types       JSONB DEFAULT '["moneyline","spread","totals"]',
    risk_tolerance  VARCHAR(20) DEFAULT 'moderate'
                    CHECK (risk_tolerance IN ('conservative','moderate','aggressive')),
    bankroll_amount DECIMAL(12,2) DEFAULT 1000.00,
    bankroll_method VARCHAR(30) DEFAULT 'fractional_kelly'
                    CHECK (bankroll_method IN ('flat','full_kelly',
                           'fractional_kelly','conservative')),
    kelly_fraction  DECIMAL(4,3) DEFAULT 0.250,  -- 1/4 Kelly default
    max_bet_pct     DECIMAL(4,3) DEFAULT 0.050,  -- 5% max single bet
    min_ev_threshold DECIMAL(5,3) DEFAULT 0.030,  -- 3% minimum EV
    notify_line_moves BOOLEAN DEFAULT TRUE,
    notify_new_picks  BOOLEAN DEFAULT TRUE,
    UNIQUE(user_id)
);

-- ============================================================
-- SPORTS DATA: GAMES & TEAMS
-- ============================================================

CREATE TABLE sports (
    id      VARCHAR(20) PRIMARY KEY,
    name    VARCHAR(100) NOT NULL,
    active  BOOLEAN DEFAULT TRUE
);

CREATE TABLE leagues (
    id          VARCHAR(50) PRIMARY KEY,
    sport_id    VARCHAR(20) REFERENCES sports(id),
    name        VARCHAR(100) NOT NULL,
    country     VARCHAR(50),
    active      BOOLEAN DEFAULT TRUE
);

CREATE TABLE teams (
    id              VARCHAR(50) PRIMARY KEY,
    league_id       VARCHAR(50) REFERENCES leagues(id),
    name            VARCHAR(100) NOT NULL,
    abbreviation    VARCHAR(10),
    elo_rating      DECIMAL(8,2),
    elo_updated_at  TIMESTAMPTZ,
    metadata        JSONB DEFAULT '{}'
);

CREATE TABLE games (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id     VARCHAR(100),
    league_id       VARCHAR(50) REFERENCES leagues(id),
    home_team_id    VARCHAR(50) REFERENCES teams(id),
    away_team_id    VARCHAR(50) REFERENCES teams(id),
    scheduled_at    TIMESTAMPTZ NOT NULL,
    status          VARCHAR(20) DEFAULT 'scheduled'
                    CHECK (status IN ('scheduled','live','final','postponed','canceled')),
    home_score      INTEGER,
    away_score      INTEGER,
    context         JSONB DEFAULT '{}',
    -- context stores: rest_days, b2b flags, weather, injuries, venue type
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(external_id)
);

CREATE INDEX idx_games_scheduled ON games(scheduled_at);
CREATE INDEX idx_games_league_status ON games(league_id, status);

-- ============================================================
-- ODDS & LINE MOVEMENT
-- ============================================================

CREATE TABLE odds_snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id         UUID REFERENCES games(id) ON DELETE CASCADE,
    sportsbook      VARCHAR(50) NOT NULL,
    market_type     VARCHAR(30) NOT NULL
                    CHECK (market_type IN ('h2h','spreads','totals',
                           'player_prop','team_total','btts','draw_no_bet')),
    home_odds       DECIMAL(8,3),
    away_odds       DECIMAL(8,3),
    draw_odds       DECIMAL(8,3),
    home_spread     DECIMAL(5,2),
    home_spread_odds DECIMAL(8,3),
    away_spread     DECIMAL(5,2),
    away_spread_odds DECIMAL(8,3),
    total_line      DECIMAL(6,2),
    over_odds       DECIMAL(8,3),
    under_odds      DECIMAL(8,3),
    prop_description VARCHAR(200),
    prop_line       DECIMAL(6,2),
    prop_over_odds  DECIMAL(8,3),
    prop_under_odds DECIMAL(8,3),
    recorded_at     TIMESTAMPTZ DEFAULT NOW(),
    is_closing_line BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_odds_game_market ON odds_snapshots(game_id, market_type);
CREATE INDEX idx_odds_recorded ON odds_snapshots(recorded_at);

-- Materialized view: no-vig consensus lines
CREATE MATERIALIZED VIEW consensus_odds AS
SELECT
    game_id, market_type,
    AVG(1.0 / home_odds) /
        (AVG(1.0 / home_odds) + AVG(1.0 / away_odds)) AS home_nv_prob,
    AVG(1.0 / away_odds) /
        (AVG(1.0 / home_odds) + AVG(1.0 / away_odds)) AS away_nv_prob,
    COUNT(DISTINCT sportsbook) AS num_books,
    MAX(recorded_at) AS last_updated
FROM odds_snapshots
WHERE market_type = 'h2h'
  AND recorded_at > NOW() - INTERVAL '30 minutes'
GROUP BY game_id, market_type;

-- ============================================================
-- MODEL PREDICTIONS
-- ============================================================

CREATE TABLE predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id         UUID REFERENCES games(id) ON DELETE CASCADE,
    market_type     VARCHAR(30) NOT NULL,
    model_name      VARCHAR(50) NOT NULL,
    home_win_prob   DECIMAL(6,5),
    away_win_prob   DECIMAL(6,5),
    draw_prob       DECIMAL(6,5),
    predicted_spread DECIMAL(5,2),
    predicted_total  DECIMAL(6,2),
    confidence      DECIMAL(5,4),
    calibration_error DECIMAL(5,4),
    top_features    JSONB DEFAULT '[]',
    -- Example: [{"feature": "home_elo_diff", "value": 85, "impact": 0.12}]
    predicted_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(game_id, market_type, model_name)
);

-- ============================================================
-- BET RECOMMENDATIONS & TRACKING
-- ============================================================

CREATE TABLE bet_recommendations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id         UUID REFERENCES games(id),
    user_id         UUID REFERENCES users(id),
    market_type     VARCHAR(30) NOT NULL,
    bet_side        VARCHAR(30) NOT NULL,
    recommended_book VARCHAR(50),
    recommended_odds DECIMAL(8,3),
    model_prob      DECIMAL(6,5) NOT NULL,
    implied_prob    DECIMAL(6,5) NOT NULL,
    ev_pct          DECIMAL(6,4) NOT NULL,
    kelly_fraction  DECIMAL(6,5),
    recommended_units DECIMAL(6,3),
    recommended_amount DECIMAL(12,2),
    edge_type       VARCHAR(30)
                    CHECK (edge_type IN ('model_driven','sharp_signal',
                           'steam_move','closing_line','correlated')),
    confidence_tier VARCHAR(10)
                    CHECK (confidence_tier IN ('A','B','C')),
    status          VARCHAR(20) DEFAULT 'active'
                    CHECK (status IN ('active','placed','won','lost',
                           'push','void','expired')),
    odds_at_placement DECIMAL(8,3),
    closing_odds    DECIMAL(8,3),
    clv_cents       DECIMAL(6,2),
    clv_pct         DECIMAL(6,4),
    recommended_at  TIMESTAMPTZ DEFAULT NOW(),
    placed_at       TIMESTAMPTZ,
    settled_at      TIMESTAMPTZ,
    actual_pnl      DECIMAL(12,2)
);

-- ============================================================
-- CONVERSATIONS & PERFORMANCE
-- ============================================================

CREATE TABLE conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    metadata    JSONB DEFAULT '{}'
);

CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role            VARCHAR(10) NOT NULL CHECK (role IN ('user','assistant','system')),
    content         TEXT NOT NULL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE model_performance (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name      VARCHAR(50) NOT NULL,
    league_id       VARCHAR(50),
    market_type     VARCHAR(30),
    eval_start      DATE NOT NULL,
    eval_end        DATE NOT NULL,
    num_predictions INTEGER,
    brier_score     DECIMAL(8,6),
    log_loss        DECIMAL(8,6),
    auc_roc         DECIMAL(6,5),
    calibration_err DECIMAL(6,5),
    num_bets        INTEGER,
    win_rate        DECIMAL(5,4),
    avg_ev          DECIMAL(6,4),
    roi             DECIMAL(6,4),
    avg_clv         DECIMAL(6,4),
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);
```

### Redis Key Structure

```
# Cached odds (TTL: 15 minutes)
odds:{game_id}:{market_type}              -> JSON blob of latest odds
odds:best:{game_id}:{market_type}:{side}  -> Best available odds

# User session
session:{user_id}                         -> JSON with conversation_id, prefs
chat:{conversation_id}:messages           -> Last 20 messages

# Rate limiting
ratelimit:odds_api:remaining              -> Integer countdown from 500
ratelimit:user:{user_id}:minute           -> Per-user request counter

# Real-time alerts
channel:line_moves                        -> Pub/Sub for WebSocket broadcast
channel:new_picks:{user_id}               -> Per-user pick alerts

# Model cache (TTL: 5 minutes)
pred:{game_id}:{model_name}               -> JSON prediction blob

# Feature cache (TTL: 1 hour)
features:{team_id}:rolling                -> Rolling team features
features:injuries:{league_id}             -> Current injury report
```

---

## 4. Data Pipeline

### Pipeline Architecture

```
+=======================================================================+
|                    SCHEDULED INGESTION JOBS (Celery Beat)              |
|                                                                       |
|  +------------------+  +------------------+  +-------------------+    |
|  | ODDS INGESTION   |  | STATS INGESTION  |  | CONTEXT INGESTION |   |
|  | (Every 6 hours)  |  | (Daily @ 4 AM)   |  | (Hourly)          |   |
|  |                  |  |                  |  |                    |   |
|  | the_odds_api     |  | nba_api          |  | injury scraper     |   |
|  | ~2-3 req/call    |  | nfl_data_py      |  | weather (Open-     |   |
|  | Budget: ~4 calls |  | pybaseball       |  |   Meteo)           |   |
|  |   per day =      |  | hockey_scraper   |  | rest day calc      |   |
|  |   ~120/month     |  | football-data    |  | travel distance    |   |
|  +--------+---------+  +--------+---------+  +--------+----------+   |
+=======================================================================+
            |                      |                      |
            v                      v                      v
+=======================================================================+
|                    VALIDATION & NORMALIZATION LAYER                    |
|                                                                       |
|  DataValidator:                                                       |
|  - Schema validation (Pydantic models)                                |
|  - Range checks (odds between 1.01 and 500.0)                        |
|  - Staleness detection (reject if >2hr old for live odds)             |
|  - Duplicate detection (idempotent upserts)                           |
|  - Cross-source reconciliation (flag >5% disagreement)               |
|                                                                       |
|  OddsNormalizer:                                                      |
|  - American <-> Decimal <-> Implied Probability conversions           |
|  - Vig removal (multiplicative method per book)                       |
|  - Consensus line calculation (weighted by book sharpness)            |
|  - Pinnacle-weighted no-vig as "true" market probability              |
|                                                                       |
|  StatsNormalizer:                                                     |
|  - Per-100-possessions normalization (NBA)                            |
|  - Per-600-minutes normalization (soccer)                             |
|  - Era-adjusted stats (MLB park factors, rule changes)                |
|  - Z-score standardization across leagues                             |
+=======================================================================+
            |
            v
+=======================================================================+
|                     FEATURE ENGINEERING PIPELINE                      |
|                                                                       |
|  TEAM-LEVEL:                                                          |
|  - Elo rating (with HCA adjustment)                                   |
|  - Rolling N-game metrics (5g, 10g, 20g windows)                      |
|  - Offensive/Defensive efficiency ratings                             |
|  - Pace/tempo, Home/Away splits, H2H history                         |
|  - Rest advantage, Travel distance                                    |
|                                                                       |
|  PLAYER-LEVEL:                                                        |
|  - Key player availability (weighted by WAR/VORP)                     |
|  - Usage rate / minutes share                                         |
|  - Matchup-specific stats                                             |
|                                                                       |
|  MARKET-LEVEL:                                                        |
|  - Opening line, Current line, Movement direction & magnitude         |
|  - Reverse Line Movement (RLM) flag                                   |
|  - Steam move detection (rapid synchronized moves)                    |
|  - Public betting % proxy                                             |
|                                                                       |
|  CONTEXTUAL:                                                          |
|  - Weather (outdoor sports: temp, wind, precip)                       |
|  - Altitude, Surface type, Timezone disadvantage                      |
|  - Scheduling spot (sandwich, trap, rivalry)                          |
+=======================================================================+
```

### The Odds API Budget Strategy (500 req/month)

```
1. Primary allocation: ~360 requests/month
   - NBA: 2 req/day x 30 = 60
   - NFL: 3 req/week x 4.5 = 14 (game days only)
   - MLB: 2 req/day x 30 = 60
   - NHL: 2 req/day x 30 = 60
   - Soccer (EPL + top 3): 1 req/day x 30 x 4 = 120
   - Buffer: ~46 requests for extras

2. Conservation strategies:
   - Only poll today's + tomorrow's games
   - Single request per sport returns all games
   - Cache aggressively: 6-hour TTL for non-game-day odds
   - Track remaining quota in Redis, alert at 80%

3. Supplemental (free):
   - Scrape free odds comparison sites as fallback
   - Cache closing lines from previous similar matchups
```

### Odds Ingestion Worker

```python
# /backend/workers/odds_ingestion.py

from celery import Celery
from pydantic import BaseModel, validator
import httpx, redis

SHARPNESS_WEIGHTS = {
    "pinnacle": 1.0, "betfair_ex_eu": 0.95, "matchbook": 0.85,
    "bookmaker": 0.80, "bovada": 0.60, "draftkings": 0.55,
    "fanduel": 0.50, "betmgm": 0.45, "pointsbet": 0.40,
}

class OddsSnapshot(BaseModel):
    game_external_id: str
    sportsbook: str
    market_type: str
    home_odds: float | None = None
    away_odds: float | None = None
    draw_odds: float | None = None
    recorded_at: datetime

    @validator("home_odds", "away_odds", "draw_odds", pre=True)
    def validate_odds_range(cls, v):
        if v is not None and (v < 1.01 or v > 500.0):
            raise ValueError(f"Odds {v} outside valid range")
        return v


def compute_no_vig_probs(home_dec, away_dec, draw_dec=None):
    """Remove vig using multiplicative method."""
    imp_home = 1.0 / home_dec
    imp_away = 1.0 / away_dec
    imp_draw = (1.0 / draw_dec) if draw_dec else 0.0
    total = imp_home + imp_away + imp_draw
    return {
        "home": imp_home / total,
        "away": imp_away / total,
        "draw": (imp_draw / total) if draw_dec else None,
        "overround": total - 1.0,
    }


def compute_consensus_line(snapshots):
    """Sharpness-weighted consensus no-vig probability."""
    weighted_home, weighted_away, weight_sum = 0.0, 0.0, 0.0
    for snap in snapshots:
        w = SHARPNESS_WEIGHTS.get(snap.sportsbook, 0.3)
        nv = compute_no_vig_probs(snap.home_odds, snap.away_odds, snap.draw_odds)
        weighted_home += nv["home"] * w
        weighted_away += nv["away"] * w
        weight_sum += w
    total = (weighted_home + weighted_away) / weight_sum
    return {
        "consensus_home_prob": (weighted_home / weight_sum) / total,
        "consensus_away_prob": (weighted_away / weight_sum) / total,
    }


@app.task(name="ingest_odds")
def ingest_odds(sport_key, markets="h2h"):
    """Fetch odds and track API budget."""
    remaining = int(r.get("ratelimit:odds_api:remaining") or 500)
    if remaining < 20:
        log.warning(f"Budget critically low: {remaining}")
        return

    response = httpx.get(
        f"{ODDS_API_BASE}/{sport_key}/odds",
        params={"apiKey": KEY, "regions": "us",
                "markets": markets, "oddsFormat": "decimal"},
    )
    new_remaining = int(response.headers.get("x-requests-remaining", remaining))
    r.set("ratelimit:odds_api:remaining", new_remaining)

    for game in response.json():
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                snapshot = OddsSnapshot(
                    game_external_id=game["id"],
                    sportsbook=bookmaker["key"],
                    market_type=market["key"],
                    home_odds=outcomes.get(game["home_team"]),
                    away_odds=outcomes.get(game["away_team"]),
                    draw_odds=outcomes.get("Draw"),
                    recorded_at=datetime.utcnow(),
                )
                save_odds_snapshot(snapshot)
        cache_latest_odds(game["id"], game)
```

### Data Conflict Resolution

```python
# /backend/pipeline/conflict_resolution.py

class DataConflictResolver:
    SOURCE_RELIABILITY = {
        "pinnacle": 0.95, "the_odds_api": 0.90,
        "nba_api_official": 0.95, "nfl_data_py": 0.95,
        "pybaseball_statcast": 0.95, "football_data_org": 0.85,
        "web_scrape": 0.60,
    }

    def resolve_numeric(self, values: list[tuple[str, float]]) -> float:
        """Reliability-weighted average for numeric conflicts."""
        total_w, weighted_sum = 0.0, 0.0
        for source, val in values:
            w = self.SOURCE_RELIABILITY.get(source, 0.5)
            weighted_sum += val * w
            total_w += w
        return weighted_sum / total_w

    def resolve_categorical(self, values: list[tuple[str, str]]) -> str:
        """Take value from highest-priority source."""
        values.sort(key=lambda x: self.SOURCE_RELIABILITY.get(x[0], 0), reverse=True)
        return values[0][1]

    def detect_anomaly(self, values, threshold=0.05) -> bool:
        """Flag if sources disagree by more than threshold."""
        nums = [v for _, v in values]
        return (max(nums) - min(nums)) > threshold
```

---

## 5. Modeling Architecture

### Layer 1: Elo Rating System (All Sports)

```python
# /backend/models/elo.py

class EloRatingSystem:
    SPORT_CONFIG = {
        "nba":    {"k": 20, "hca": 100, "mean": 1500, "regress_pct": 0.25},
        "nfl":    {"k": 20, "hca": 48,  "mean": 1500, "regress_pct": 0.33},
        "mlb":    {"k": 4,  "hca": 24,  "mean": 1500, "regress_pct": 0.40},
        "nhl":    {"k": 6,  "hca": 30,  "mean": 1500, "regress_pct": 0.30},
        "soccer": {"k": 25, "hca": 75,  "mean": 1500, "regress_pct": 0.20},
    }

    def __init__(self, sport):
        self.config = self.SPORT_CONFIG[sport]
        self.ratings = {}

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def predict_game(self, home_id, away_id):
        home_elo = self.ratings.get(home_id, self.config["mean"])
        away_elo = self.ratings.get(away_id, self.config["mean"])
        adjusted_home = home_elo + self.config["hca"]
        home_win_prob = self.expected_score(adjusted_home, away_elo)
        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": 1.0 - home_win_prob,
            "elo_diff": adjusted_home - away_elo,
        }

    def update_ratings(self, home_id, away_id, home_score, away_score):
        pred = self.predict_game(home_id, away_id)
        actual_home = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)

        # Margin-of-victory multiplier
        score_diff = abs(home_score - away_score)
        mov_mult = math.log(max(score_diff, 1) + 1) * (
            2.2 / ((pred["elo_diff"] * 0.001) + 2.2)
        )
        k = self.config["k"] * mov_mult

        home_elo = self.ratings.get(home_id, self.config["mean"])
        away_elo = self.ratings.get(away_id, self.config["mean"])
        self.ratings[home_id] = home_elo + k * (actual_home - pred["home_win_prob"])
        self.ratings[away_id] = away_elo + k * ((1 - actual_home) - pred["away_win_prob"])

    def season_regression(self):
        mean, pct = self.config["mean"], self.config["regress_pct"]
        for tid in self.ratings:
            self.ratings[tid] = self.ratings[tid] * (1 - pct) + mean * pct
```

### Layer 1b: Poisson Scoring Model (Soccer, NHL, MLB)

```python
# /backend/models/poisson.py

import numpy as np
from scipy.stats import poisson

class PoissonScoringModel:
    """Models scoring as independent Poisson processes."""

    def fit(self, game_history):
        """Compute per-team attack/defense ratings (simplified Dixon-Coles)."""
        team_gf, team_ga, team_games = {}, {}, {}
        total_goals, total_games = 0, 0

        for game in game_history:
            for side in ['home', 'away']:
                tid = game[f'{side}_team_id']
                opp = 'away' if side == 'home' else 'home'
                gf, ga = game[f'{side}_score'], game[f'{opp}_score']
                team_gf.setdefault(tid, 0); team_ga.setdefault(tid, 0)
                team_games.setdefault(tid, 0)
                team_gf[tid] += gf; team_ga[tid] += ga; team_games[tid] += 1
            total_goals += game['home_score'] + game['away_score']
            total_games += 1

        self.league_avg = total_goals / (2 * total_games)
        self.attack = {t: team_gf[t] / team_games[t] / self.league_avg for t in team_games}
        self.defense = {t: team_ga[t] / team_games[t] / self.league_avg for t in team_games}

    def predict_game(self, home_id, away_id, hca=1.25):
        lambda_home = self.attack.get(home_id, 1.0) * self.defense.get(away_id, 1.0) * self.league_avg * hca
        lambda_away = self.attack.get(away_id, 1.0) * self.defense.get(home_id, 1.0) * self.league_avg

        # Score probability matrix (up to 10 goals)
        score_matrix = np.zeros((11, 11))
        for i in range(11):
            for j in range(11):
                score_matrix[i][j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)

        return {
            "home_win_prob": float(np.sum(np.tril(score_matrix, -1))),
            "draw_prob": float(np.sum(np.diag(score_matrix))),
            "away_win_prob": float(np.sum(np.triu(score_matrix, 1))),
            "expected_total": float(lambda_home + lambda_away),
        }
```

### Layer 2: ML Ensemble (XGBoost + LightGBM + Random Forest)

```python
# /backend/models/ml_ensemble.py

class SportsBettingMLPipeline:
    """
    Feature vector example (NBA):
    [elo_diff, home_off_rtg_10g, home_def_rtg_10g, away_off_rtg_10g,
     away_def_rtg_10g, home_pace_10g, away_pace_10g, home_rest_days,
     away_rest_days, home_b2b, away_b2b, home_win_pct_last20,
     away_win_pct_last20, home_away_split_diff, injury_impact_home,
     injury_impact_away, h2h_home_win_pct, market_implied_home,
     line_movement_dir, is_rlm]
    """

    def __init__(self, sport):
        self.sport = sport
        self.models = {
            "xgboost": xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss"),
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                reg_alpha=0.1, reg_lambda=1.0, verbose=-1),
            "random_forest": RandomForestClassifier(
                n_estimators=500, max_depth=8, min_samples_leaf=20,
                max_features="sqrt", n_jobs=-1),
        }
        self.calibrated_models = {}
        self.ensemble_weights = {"xgboost": 0.45, "lightgbm": 0.35, "random_forest": 0.20}

    def train(self, X, y):
        """Train with time-series CV. NEVER use future data to predict past."""
        tscv = TimeSeriesSplit(n_splits=5)
        for name, model in self.models.items():
            calibrated = CalibratedClassifierCV(model, cv=tscv, method="isotonic")
            calibrated.fit(X[self.features], y)
            self.calibrated_models[name] = calibrated
        self._optimize_weights(X, y, tscv)

    def _optimize_weights(self, X, y, tscv):
        """Optimize ensemble weights via log-loss minimization on validation fold."""
        train_idx, val_idx = list(tscv.split(X))[-1]
        X_val, y_val = X.iloc[val_idx][self.features], y.iloc[val_idx]
        model_probs = {name: m.predict_proba(X_val)[:, 1]
                       for name, m in self.calibrated_models.items()}

        def neg_log_loss(weights):
            w = np.array(weights); w = w / w.sum()
            combined = np.clip(sum(w[i] * model_probs[n]
                       for i, n in enumerate(model_probs)), 1e-7, 1-1e-7)
            return -np.mean(y_val * np.log(combined) + (1-y_val) * np.log(1-combined))

        result = minimize(neg_log_loss, x0=[1/3]*3, method="Nelder-Mead")
        names = list(model_probs.keys())
        opt = result.x / result.x.sum()
        self.ensemble_weights = {names[i]: opt[i] for i in range(len(names))}

    def predict(self, X):
        probs = {n: m.predict_proba(X[self.features])[:, 1]
                 for n, m in self.calibrated_models.items()}
        ensemble_prob = sum(self.ensemble_weights[n] * probs[n] for n in probs)

        importances = self.calibrated_models["xgboost"].estimator.feature_importances_
        top_features = sorted(zip(self.features, importances),
                              key=lambda x: x[1], reverse=True)[:5]
        return {
            "ensemble_prob": float(ensemble_prob[0]),
            "model_probs": {n: float(probs[n][0]) for n in probs},
            "top_features": [{"feature": f, "importance": float(imp)}
                             for f, imp in top_features],
        }
```

### Layer 3: Bayesian Updating

```python
# /backend/models/bayesian.py

from scipy.stats import beta as beta_dist

class BayesianProbabilityUpdater:
    """
    Maintains Beta posteriors for team win probabilities.
    Key insight: ML gives point estimates; Bayesian gives DISTRIBUTIONS,
    letting us quantify uncertainty and size bets more conservatively.
    """

    def __init__(self):
        self.posteriors = {}  # team_id -> (alpha, beta)

    def initialize_from_elo(self, team_id, elo_win_prob, equiv_games=20):
        """Convert Elo probability to Beta prior.
        60% win rate -> Beta(12, 8), mean=0.60 with moderate uncertainty."""
        self.posteriors[team_id] = (elo_win_prob * equiv_games,
                                     (1 - elo_win_prob) * equiv_games)

    def update_with_result(self, team_id, won: bool):
        alpha, b = self.posteriors.get(team_id, (10, 10))
        self.posteriors[team_id] = (alpha + 1, b) if won else (alpha, b + 1)

    def update_with_evidence(self, team_id, strength, direction="positive"):
        """Partial update for non-game evidence (e.g., star player OUT).
        update_with_evidence(team_id, 0.3, "negative") adds 0.3 to beta."""
        alpha, b = self.posteriors.get(team_id, (10, 10))
        if direction == "positive":
            self.posteriors[team_id] = (alpha + strength, b)
        else:
            self.posteriors[team_id] = (alpha, b + strength)

    def get_matchup_probability(self, home_id, away_id):
        """Monte Carlo from each team's Beta posterior."""
        h_a, h_b = self.posteriors.get(home_id, (10, 10))
        a_a, a_b = self.posteriors.get(away_id, (10, 10))
        home_samples = np.random.beta(h_a, h_b, 10000)
        away_samples = np.random.beta(a_a, a_b, 10000)
        win_samples = home_samples / (home_samples + away_samples)
        return {
            "home_win_prob": float(np.mean(win_samples)),
            "uncertainty": float(np.std(win_samples)),
            "ci_90": (float(np.percentile(win_samples, 5)),
                      float(np.percentile(win_samples, 95))),
        }
```

---

## 6. Bet Selection Engine

### EV Calculator & Bet Ranker

```python
# /backend/engine/ev_calculator.py

class EVCalculator:
    """Combines models, calculates EV, sizes bets, ranks candidates."""

    def compute_fair_probability(self, game):
        """
        Ensemble weights (sport-specific, optimizable via backtesting):
        - ML ensemble: 40%    (highest predictive accuracy)
        - Market consensus: 25% (efficient pricing)
        - Bayesian posterior: 20% (uncertainty quantification)
        - Elo baseline: 15%   (stable, low-variance)
        """
        elo = self.elo_model.predict_game(game["home_team_id"], game["away_team_id"])
        ml = self.ml_pipeline.predict(game["features"])
        bayes = self.bayesian_updater.get_matchup_probability(
            game["home_team_id"], game["away_team_id"])
        market = game.get("consensus_no_vig", {})

        weights = self._get_sport_weights(game["sport"])
        fair_home = (weights["ml"] * ml["ensemble_prob"] +
                     weights["market"] * market.get("home_prob", 0.5) +
                     weights["bayesian"] * bayes["home_win_prob"] +
                     weights["elo"] * elo["home_win_prob"])

        return {
            "fair_home_prob": fair_home,
            "fair_away_prob": 1.0 - fair_home,
            "uncertainty": bayes["uncertainty"],
            "model_breakdown": {
                "elo": elo["home_win_prob"],
                "ml_ensemble": ml["ensemble_prob"],
                "bayesian": bayes["home_win_prob"],
                "market": market.get("home_prob"),
            },
            "top_features": ml["top_features"],
        }

    def calculate_ev(self, fair_prob, decimal_odds):
        """
        EV% = (fair_prob * decimal_odds) - 1

        Example: fair_prob=0.55, odds=2.10 (+110)
        EV = (0.55 * 2.10) - 1 = +15.5%
        """
        ev = (fair_prob * decimal_odds) - 1.0
        implied = 1.0 / decimal_odds
        return {
            "ev_pct": ev,
            "edge_pct": fair_prob - implied,
            "fair_prob": fair_prob,
            "implied_prob": implied,
            "is_positive_ev": ev > 0,
        }

    def compute_kelly_stake(self, fair_prob, decimal_odds, bankroll,
                             kelly_frac=0.25, max_bet_pct=0.05):
        """
        Full Kelly: f = (bp - q) / b
        Applied: 1/4 Kelly with 5% bankroll cap.
        """
        b = decimal_odds - 1.0
        full_kelly = (b * fair_prob - (1 - fair_prob)) / b
        adjusted = max(0.0, min(full_kelly * kelly_frac, max_bet_pct))
        return {
            "full_kelly_fraction": full_kelly,
            "final_bet_fraction": adjusted,
            "recommended_amount": round(bankroll * adjusted, 2),
            "recommended_units": round(adjusted * 100, 2),
        }

    def rank_bets(self, candidates, user_prefs):
        """Rank by composite: EV (50%) + Confidence (25%) +
        Uncertainty penalty (15%) + Correlation penalty (10%)."""
        min_ev = user_prefs.get("min_ev_threshold", 0.03)
        scored = []
        for bet in candidates:
            if bet["ev_pct"] < min_ev:
                continue
            composite = (
                bet["ev_pct"] * 100 * 0.50 +
                bet.get("model_agreement", 0.5) * 10 * 0.25 +
                bet.get("uncertainty", 0.1) * -20 * 0.15 +
                self._correlation_penalty(bet, scored) * 0.10
            )
            bet["composite_score"] = composite
            bet["confidence_tier"] = "A" if composite > 8 else ("B" if composite > 4 else "C")
            scored.append(bet)
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        return scored

    def classify_edge_type(self, bet):
        """Classify: steam_move | sharp_signal | closing_line | model_driven."""
        if bet.get("steam_move"): return "steam_move"
        if bet.get("is_reverse_line_movement") and bet["edge_pct"] > 0.03: return "sharp_signal"
        if bet.get("clv_historical", 0) > 0.02: return "closing_line"
        return "model_driven"
```

### Line Movement & Sharp Money Detector

```python
# /backend/engine/line_movement.py

class LineMovementAnalyzer:
    """Detect RLM, steam moves, and line velocity."""

    def analyze_movement(self, game_id, odds_history):
        if len(odds_history) < 2:
            return {"has_movement": False}

        opening, current = odds_history[0], odds_history[-1]
        prob_shift = 1.0/current["home_odds"] - 1.0/opening["home_odds"]

        return {
            "has_movement": abs(prob_shift) > 0.01,
            "prob_shift": prob_shift,
            "direction": "toward_home" if prob_shift > 0 else "toward_away",
            "magnitude_cents": abs(prob_shift) * 100,
            "is_steam_move": self._detect_steam(odds_history)["detected"],
            "is_reverse_line_movement": self._detect_rlm(odds_history, prob_shift),
            "velocity": self._compute_velocity(odds_history),
        }

    def _detect_steam(self, history):
        """Steam = 3+ books move same direction within 15-min window."""
        # Group by 15-min windows, check synchronized moves
        # Returns {"detected": bool, "books": [...]}
        ...

    def _detect_rlm(self, history, prob_shift):
        """RLM: line moves AGAINST the popular side (proxy: favorites)."""
        opening = history[0]
        home_was_fav = opening["home_odds"] < opening["away_odds"]
        if home_was_fav and prob_shift < -0.02: return True
        if not home_was_fav and prob_shift > 0.02: return True
        return False
```

---

## 7. API Design

### RESTful Endpoints (FastAPI)

```
Base URL: /api/v1 | Auth: Bearer JWT

CHAT:
  POST /chat/message          Send message (SSE streaming response)
  GET  /chat/conversations    List user conversations
  GET  /chat/conversations/{id}/messages

PICKS:
  GET  /picks/today           Today's recommended bets
    ?sport=nba|nfl|mlb|nhl|soccer|all
    ?min_ev=0.03
    ?tier=A|B|C|all
    ?market=moneyline|spread|total|all

GAMES:
  GET  /games/today           All scheduled games
  GET  /games/{id}            Game detail with odds + predictions
  GET  /games/{id}/odds       Odds history, best lines
  GET  /games/{id}/prediction Model prediction breakdown

USER:
  GET  /user/preferences      Get betting preferences
  PUT  /user/preferences      Update preferences
  GET  /user/performance      ROI, CLV, win rate, P&L

ANALYTICS:
  GET  /analytics/model-performance   Brier scores, calibration
  GET  /analytics/clv-tracker         CLV across settled bets
  GET  /analytics/bankroll-history    Growth chart data

WEBSOCKET:
  WS   /ws/live               Real-time line moves, new picks, injuries
```

### Example Picks Response

```json
{
  "picks": [{
    "id": "uuid",
    "game": {
      "home_team": "Boston Celtics",
      "away_team": "New York Knicks",
      "scheduled_at": "2026-02-07T19:30:00Z",
      "league": "NBA"
    },
    "recommendation": {
      "side": "home_ml",
      "display": "Celtics ML",
      "best_odds": -145,
      "best_book": "FanDuel",
      "fair_prob": 0.628,
      "implied_prob": 0.592,
      "ev_pct": 0.061,
      "edge_type": "model_driven",
      "confidence_tier": "A",
      "kelly_units": 1.8,
      "recommended_amount": 45.00
    },
    "reasoning": {
      "summary": "Model gives Celtics 62.8% vs market's 59.2%",
      "top_factors": [
        "Celtics +85 Elo advantage at home",
        "Knicks on second of back-to-back",
        "Celtics 8-2 in last 10 home games"
      ],
      "model_breakdown": {
        "elo": 0.615, "ml_ensemble": 0.641,
        "bayesian": 0.622, "market": 0.605
      }
    }
  }]
}
```

---

## 8. Chatbot Prompt Engineering

### System Prompt

```xml
<system_prompt>
<role>
You are EDGE, a professional-grade sports betting analyst powered by
statistical models, machine learning, and Bayesian probability theory.
You help users identify positive expected value (+EV) betting opportunities
with full transparency about reasoning, confidence, and limitations.
</role>

<personality>
- Analytical and precise, like a quant at a trading desk
- Honest about uncertainty -- always state confidence levels
- Never hype or oversell -- if a pick is marginal, say so
- Use specific numbers, not vague qualifiers
</personality>

<constraints>
- NEVER guarantee outcomes or imply certainty
- ALWAYS include confidence tier (A/B/C) and EV percentage
- ALWAYS mention best available odds and which sportsbook
- NEVER recommend a bet with negative expected value
- If no +EV plays exist, say "No actionable edges found today"
- Include responsible gambling reminder in first session message
</constraints>

<output_format>
**[CONFIDENCE_TIER] [TEAM/SIDE] [BET_TYPE] @ [ODDS] ([SPORTSBOOK])**
- **EV:** +X.X% | **Edge:** X.X% | **Fair Prob:** XX.X%
- **Key Factors:**
  1. [Most important factor with specific stat]
  2. [Second factor]
  3. [Third factor]
- **Model Breakdown:** Elo: XX% | ML: XX% | Bayesian: XX% | Market: XX%
- **Sizing:** X.X units (based on your [bankroll_method] settings)
- **Risk Note:** [Caveats -- small sample, injury uncertainty, etc.]
</output_format>

<uncertainty_rule>
If models disagree by >5pp or Bayesian CI is wider than 15pp, flag it:
"Note: Unusual model disagreement on this game. ML says X% but Bayesian
gives Y%-Z%. Recommend reduced sizing or pass."
</uncertainty_rule>

<responsible_gambling>
If user mentions chasing losses, betting beyond means, or asks for "locks":
respond with empathy, recommend discipline, and provide:
National Council on Problem Gambling: 1-800-522-4700 | ncpgambling.org
</responsible_gambling>
</system_prompt>
```

### Context Injection Template (Per Message)

```xml
<context>
<user_profile>
  <risk_tolerance>{{risk_tolerance}}</risk_tolerance>
  <bankroll>{{bankroll_amount}}</bankroll>
  <bankroll_method>{{bankroll_method}}</bankroll_method>
  <kelly_fraction>{{kelly_fraction}}</kelly_fraction>
  <min_ev>{{min_ev_threshold}}</min_ev>
  <enabled_sports>{{enabled_sports}}</enabled_sports>
</user_profile>

<todays_picks>
{{#each picks}}
<pick confidence="{{confidence_tier}}" ev="{{ev_pct}}">
  <game>{{away_team}} @ {{home_team}} | {{scheduled_at}}</game>
  <recommendation>{{display}} @ {{best_odds}} ({{best_book}})</recommendation>
  <fair_prob>{{fair_prob}}</fair_prob>
  <ev_pct>{{ev_pct}}</ev_pct>
  <edge_type>{{edge_type}}</edge_type>
  <kelly_units>{{kelly_units}}</kelly_units>
  <top_factors>{{top_factors}}</top_factors>
  <model_breakdown>{{model_breakdown}}</model_breakdown>
</pick>
{{/each}}
</todays_picks>

<odds_freshness>{{odds_last_updated}}</odds_freshness>
</context>
```

### Token Optimization Strategy
- Compress conversation history with summaries after 10 messages
- Use Haiku for classification ("is this a betting question or general chat?")
- Reserve Sonnet for analysis explanations
- Cache static context (user profile, model descriptions)

---

## 9. Example Chatbot Responses

### Example 1: "What's your best play tonight?"

**User:** What are your top picks for tonight?

**EDGE:** Good evening. Here are tonight's top plays (odds as of 2:15 PM ET):

---

**[A] Celtics ML @ -145 (FanDuel)**
- **EV:** +6.1% | **Edge:** 3.6% | **Fair Prob:** 62.8%
- **Key Factors:**
  1. Celtics +85 Elo advantage at home; 8-2 in last 10 at TD Garden
  2. Knicks on second night of a back-to-back (played in Chicago last night)
  3. Knicks missing Mitchell Robinson (rim protection drops 4.2 pts/game without him per 3-year sample)
- **Model Breakdown:** Elo: 61.5% | ML: 64.1% | Bayesian: 62.2% | Market: 60.5%
- **Sizing:** 1.8 units ($45.00 based on your 1/4 Kelly settings)

---

**[B] Rangers/Sabres UNDER 5.5 @ -110 (DraftKings)**
- **EV:** +3.8% | **Edge:** 2.1% | **Fair Prob:** 55.2%
- **Key Factors:**
  1. Poisson model projects 4.8 total goals (home: 2.7, away: 2.1)
  2. Rangers allowing league-lowest 2.3 GA/game at home in last 15
  3. Both starting goalies posting .925+ SV% in last 10 starts
- **Model Breakdown:** ML: 56.1% | Poisson: 54.3% | Market: 53.8%
- **Sizing:** 1.2 units ($30.00)
- **Risk Note:** Sabres on 4-game losing streak which could lead to lineup changes. Monitor pregame.

---

No other games clear your 3% EV threshold tonight. The Lakers-Warriors game is tempting but our models disagree (ML: 54% LAL, Bayesian: 48% LAL) so I'm passing.

*Even +EV bets lose regularly. Bankroll management and discipline are what separate profitable bettors from recreational ones.*

---

### Example 2: "Why do you like the Celtics?"

**User:** Break down the Celtics pick more. Why not just take the spread?

**EDGE:** Let me dig deeper.

**Why moneyline over spread (-6.5):**

1. Our spread model projects Celtics by 5.2 points. The -6.5 spread is ABOVE our projection, meaning it has -1.3% EV. The market has already priced in most of the advantage.
2. The moneyline at -145 implies 59.2%. Our ensemble says 62.8%. That 3.6pp gap is where the value lives.

**Top-5 SHAP features for this game:**
1. **away_b2b = TRUE** (+4.2% toward Celtics). B2B teams cover only 44% over last 5 seasons.
2. **elo_diff = +85** (+3.1%). Significant talent gap.
3. **injury_impact_away = 0.92** (+2.8%). Robinson's absence announced 2 hours ago -- not fully priced in.
4. **home_def_rtg_10g = 106.2** (+1.5%). Elite recent home defense.
5. **away_rest_days = 1** (+1.2%). Compounds B2B factor.

The Bayesian posterior (62.2%) is slightly lower because it weights full-season data more, and the Knicks' season numbers are solid. 90% credible interval: [56.1%, 68.4%] -- reasonably tight, models are fairly confident.

**Line movement:** Opened -140, now -145. No RLM detected. Consistent with public money on Celtics; hasn't moved far enough to suggest sharp action on the other side.

---

### Example 3: Responsible Gambling Trigger

**User:** I've lost my last 8 bets. I need a lock tonight. Going to double up to get back to even.

**EDGE:** I want to be straightforward: there is no "lock" in sports betting. Every bet carries risk. An 8-bet losing streak is statistically normal even at a 55% win rate.

More importantly, doubling your bet size to chase losses is the Martingale fallacy. It mathematically increases your risk of ruin.

Here's what I recommend instead:
1. **Stick to your preset unit size.** Your 1/4 Kelly settings exist for exactly this.
2. **Review recent bets.** Were they all +EV? If so, variance will correct. If some were marginal, tighten your EV threshold.
3. **Consider a day off.** The market will be here tomorrow.

Want me to pull up your CLV numbers for those 8 bets? If you were beating the closing line, your process is sound.

If gambling is causing stress: **1-800-522-4700** | **ncpgambling.org**

---

## 10. Deployment Architecture

### Cost-Optimized Cloud Stack

```
+=========================================================================+
|  DEPLOYMENT (Target: <$30/month total)                                  |
|                                                                         |
|  Vercel (Free)           - Next.js frontend, edge functions, HTTPS      |
|  Railway (~$5/mo)        - FastAPI + Celery in Docker, 512MB RAM        |
|  Supabase (Free)         - PostgreSQL, 500MB storage                    |
|  Upstash (Free)          - Redis, 10K commands/day                      |
|  Cloudflare R2 (Free)    - Model artifacts, 10GB                        |
|  Claude API (~$15-25/mo) - Haiku triage + Sonnet analysis               |
|  The Odds API (Free)     - 500 requests/month                           |
|  GitHub Actions (Free)   - CI/CD                                        |
|  Sentry + Grafana (Free) - Monitoring                                   |
+=========================================================================+
```

### Scaling Triggers

| Trigger | Action | Cost |
|---------|--------|------|
| >10 concurrent users | Railway Starter ($5/mo) | +$5 |
| >500MB database | Supabase Pro ($25/mo) | +$25 |
| Need more odds polling | Odds API Rookie ($20/mo, 20K req) | +$20 |
| Need live odds (<5 min) | WebSocket odds provider | +$50-100 |
| GPU for training | Modal.com pay-per-use | ~$5/run |

---

## 11. Development Roadmap

### Phase 1: Foundation (Weeks 1-3) -- "Walk"

| Week | Tasks |
|------|-------|
| 1 | Monorepo setup (Next.js + FastAPI). Supabase, Upstash, Railway config. Auth (Clerk). Deploy skeleton. Create full DB schema with Alembic migrations. |
| 2 | Odds API integration + ingestion worker. No-vig + consensus logic. Celery Beat scheduler. Stats integrations for all 5 sports (nba_api, nfl_data_py, pybaseball, hockey_scraper, football-data.org). |
| 3 | Elo system for all sports (backfill from 2020). Basic EV calculator. Claude chatbot integration with system prompt + streaming. Basic chat UI. |

**Deliverable:** Chat with EDGE, get Elo-based recommendations with live odds.

### Phase 2: Model Sophistication (Weeks 4-7) -- "Run"

| Week | Tasks |
|------|-------|
| 4 | Feature engineering pipeline (rolling stats, rest, injuries, weather). Poisson model for soccer/NHL/MLB. |
| 5 | XGBoost training (NBA primary). LightGBM + Random Forest. Calibration (isotonic regression). |
| 6 | Ensemble combiner with optimized weights. Bayesian updating layer with uncertainty quantification. |
| 7 | Sharp money detection (RLM, steam). CLV tracking. Train all 5 sports. Backtest benchmark (Brier, AUC-ROC, simulated ROI). |

**Deliverable:** Multi-model ensemble with calibrated probabilities, backtested across 3 seasons.

### Phase 3: Bet Engine & UX (Weeks 8-10) -- "Sprint"

| Week | Tasks |
|------|-------|
| 8 | Full EV calculator + Kelly sizing. Bet ranking + correlation detection. Edge classification. User preferences UI. |
| 9 | Enhanced Claude prompts (dynamic context, SHAP explanations). Multi-turn conversation state. Performance dashboard (ROI, CLV, calibration, bankroll charts). |
| 10 | WebSocket live alerts. Responsible gambling features (loss limits, session tracking, self-exclusion). Compliance text. |

**Deliverable:** Production-ready chatbot with full bet engine, user controls, and guardrails.

### Phase 4: Polish & Scale (Weeks 11-12) -- "Optimize"

| Week | Tasks |
|------|-------|
| 11 | Automated weekly retraining pipeline. A/B testing for model versions. Smart caching + Claude API cost optimization (Haiku routing). |
| 12 | Load testing, error hardening, Sentry alerts. API docs + user guide. Landing page + onboarding. Public model performance transparency page. |

### Post-Launch (Continuous)

- **Monthly:** Retrain models, recalibrate ensemble weights
- **Weekly:** Review CLV performance, adjust EV thresholds
- **Per-Season:** Elo regression, update features for rule changes
- **Ongoing:** Explore player prop models, college sports, LSTM for time-series form

---

## 12. Appendix: Key Trade-offs

### Why XGBoost/LightGBM over Neural Networks?

For sports betting with limited data (hundreds to low-thousands of games per season), gradient-boosted trees consistently outperform deep learning. Neural nets need orders-of-magnitude more data and are harder to interpret. Explainability is critical for user trust. Neural nets remain viable for player-level props where tracking data provides millions of data points.

### Why weight market consensus at 25%?

The efficient market hypothesis partially applies. Sharp closing lines (especially Pinnacle) are remarkably well-calibrated. Ignoring them is arrogant; relying on them entirely means you can never find an edge. 25% is a compromise: stabilizes predictions but allows model-driven edges to surface.

### Why 1/4 Kelly as default?

Full Kelly optimizes geometric growth but has extreme variance. A 1/4 Kelly bettor has only a 1/81 chance of halving their bankroll before doubling it (vs 1/3 for full Kelly). The reduced volatility is worth the slower growth. Power users can adjust upward.

### Why 6-hour odds caching instead of live?

With 500 free API requests/month, polling every 5 minutes for one sport exhausts the budget in 3.5 days. The 6-hour cadence with 4 sports yields ~120 requests/month, leaving buffer. Upgrade to $20/month tier (20,000 requests) for tighter polling.

---

## Critical Implementation Files (Priority Order)

1. **`/backend/engine/ev_calculator.py`** -- Core brain: EV, Kelly, ranking, ensemble combination
2. **`/backend/models/ml_ensemble.py`** -- ML pipeline: XGBoost/LightGBM/RF training + prediction
3. **`/backend/workers/odds_ingestion.py`** -- Data lifeline: Odds API + budget management
4. **`/backend/api/chat_service.py`** -- UX layer: Claude integration, context building, streaming
5. **`/frontend/app/chat/page.tsx`** -- User interface: chat, bet cards, real-time odds

---

*Built for edge. Not for hype.*
