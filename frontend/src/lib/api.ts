// Determine API base URL based on environment
const API_BASE = (() => {
  // In development, connect to backend on port 8000
  if (process.env.NODE_ENV === "development") {
    return "http://localhost:8000/api";
  }
  // In production, use relative path (assumes backend is proxied)
  return "/api";
})();

export interface PickFilters {
  sport?: string;
  tier?: string;
  min_ev?: number;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  picks_referenced?: number;
}

// Normalized Pick type used by frontend components
export interface Pick {
  id: string;
  sport: string;
  league: string;
  game: {
    home_team: string;
    away_team: string;
    scheduled_time: string;
    venue?: string;
  };
  pick: {
    type: string;
    selection: string;
    odds: number;
    line?: number;
  };
  analysis: {
    ev_percentage: number;
    edge_percentage: number;
    fair_probability: number;
    confidence_tier: "A" | "B" | "C";
    key_factors: string[];
    model_breakdown: Record<string, number>;
  };
  sizing: {
    kelly_fraction: number;
    recommended_units: number;
    recommended_amount?: number;
  };
}

export interface PicksResponse {
  picks: Pick[];
  generated_at: string;
  total_picks: number;
  filters_applied?: Record<string, string>;
}

export interface GamesResponse {
  games: Array<{
    id: string;
    sport: string;
    home_team: string;
    away_team: string;
    scheduled_time: string;
    status: string;
  }>;
}

export interface UserPreferences {
  enabled_sports: string[];
  risk_tolerance: "conservative" | "moderate" | "aggressive";
  bankroll_amount: number;
  bankroll_method: "flat" | "kelly" | "fractional_kelly";
  kelly_fraction: number;
  min_ev_threshold: number;
}

async function apiRequest<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error");
    throw new Error(`API Error ${response.status}: ${errorText}`);
  }

  return response.json();
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function normalizeRawPick(raw: any): Pick {
  // Backend format uses: game.league, game.scheduled_at, recommendation.*
  const rec = raw.recommendation || {};
  const game = raw.game || {};
  const reasoning = raw.reasoning || {};
  const models = raw.model_probabilities || reasoning.model_breakdown || reasoning.model_values || {};

  return {
    id: raw.id || "unknown",
    sport: (raw.sport || "").toUpperCase(),
    league: game.league || (raw.sport || "").toUpperCase(),
    game: {
      home_team: game.home_team || "",
      away_team: game.away_team || "",
      scheduled_time: game.scheduled_at || game.scheduled_time || "",
      venue: game.venue,
    },
    pick: {
      type: rec.side || rec.type || "moneyline",
      selection: rec.display || rec.selection || "",
      odds: rec.best_odds_american || rec.odds || 0,
      line: rec.line,
    },
    analysis: {
      ev_percentage: typeof rec.ev_pct === "number"
        ? (rec.ev_pct < 1 ? rec.ev_pct * 100 : rec.ev_pct)
        : (rec.ev_percentage || 0),
      edge_percentage: typeof rec.edge === "number"
        ? (rec.edge < 1 ? rec.edge * 100 : rec.edge)
        : (rec.fair_prob && rec.implied_prob
          ? (rec.fair_prob - rec.implied_prob) * 100
          : (rec.edge_percentage || 0)),
      fair_probability: rec.fair_prob || rec.fair_probability || 0,
      confidence_tier: rec.confidence_tier || "C",
      key_factors: reasoning.top_factors || reasoning.key_factors || [],
      model_breakdown: models,
    },
    sizing: {
      kelly_fraction: rec.kelly_fraction || 0.25,
      recommended_units: rec.kelly_units || rec.recommended_units || 0,
      recommended_amount: rec.recommended_amount,
    },
  };
}

export async function fetchPicks(
  filters?: PickFilters
): Promise<PicksResponse> {
  const params = new URLSearchParams();
  if (filters?.sport) params.set("sport", filters.sport);
  if (filters?.tier) params.set("tier", filters.tier);
  if (filters?.min_ev !== undefined)
    params.set("min_ev", filters.min_ev.toString());

  const query = params.toString();
  const endpoint = `/picks/today${query ? `?${query}` : ""}`;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await apiRequest(endpoint);

  // Normalize backend response to frontend Pick format
  const rawPicks = raw.picks || [];
  const picks: Pick[] = rawPicks.map(normalizeRawPick);
  const meta = raw.metadata || {};

  return {
    picks,
    generated_at: meta.generated_at || raw.generated_at || new Date().toISOString(),
    total_picks: meta.total_picks || raw.total_picks || picks.length,
    filters_applied: meta.filters || raw.filters_applied,
  };
}

export async function fetchGames(sport?: string): Promise<GamesResponse> {
  const params = new URLSearchParams();
  if (sport) params.set("sport", sport);

  const query = params.toString();
  const endpoint = `/games/today${query ? `?${query}` : ""}`;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await apiRequest(endpoint);

  return {
    games: (raw.games || []).map((g: Record<string, string>) => ({
      id: g.id,
      sport: g.sport,
      home_team: g.home_team,
      away_team: g.away_team,
      scheduled_time: g.scheduled_at || g.scheduled_time,
      status: g.status,
    })),
  };
}

export async function sendChatMessage(
  message: string,
  conversationId?: string | null
): Promise<ChatResponse> {
  const body: Record<string, unknown> = {
    message,
    stream: false,
  };
  if (conversationId) {
    body.conversation_id = conversationId;
  }

  return apiRequest<ChatResponse>("/chat/message", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function fetchPreferences(): Promise<UserPreferences> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await apiRequest("/users/preferences");
  // Backend wraps in { preferences: {...} }
  const p = raw.preferences || raw;
  return {
    enabled_sports: p.enabled_sports || [],
    risk_tolerance: p.risk_tolerance || "moderate",
    bankroll_amount: p.bankroll_amount || 1000,
    bankroll_method: p.bankroll_method || "fractional_kelly",
    kelly_fraction: p.kelly_fraction || 0.25,
    min_ev_threshold: typeof p.min_ev_threshold === "number"
      ? (p.min_ev_threshold < 1 ? p.min_ev_threshold * 100 : p.min_ev_threshold)
      : 3.0,
  };
}

export async function updatePreferences(
  prefs: Partial<UserPreferences>
): Promise<UserPreferences> {
  // Send threshold as decimal to backend
  const payload = { ...prefs };
  if (payload.min_ev_threshold !== undefined && payload.min_ev_threshold >= 1) {
    payload.min_ev_threshold = payload.min_ev_threshold / 100;
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await apiRequest("/users/preferences", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
  const p = raw.preferences || raw;
  return {
    enabled_sports: p.enabled_sports || [],
    risk_tolerance: p.risk_tolerance || "moderate",
    bankroll_amount: p.bankroll_amount || 1000,
    bankroll_method: p.bankroll_method || "fractional_kelly",
    kelly_fraction: p.kelly_fraction || 0.25,
    min_ev_threshold: typeof p.min_ev_threshold === "number"
      ? (p.min_ev_threshold < 1 ? p.min_ev_threshold * 100 : p.min_ev_threshold)
      : 3.0,
  };
}
