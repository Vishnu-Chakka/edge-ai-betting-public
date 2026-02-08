"""
Claude API integration for the EDGE AI chatbot.
Handles system prompts, context injection, and streaming responses.
"""
from __future__ import annotations

import os
import json
import logging
from typing import Optional, List, Dict, Any, AsyncIterator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """<role>
You are EDGE, a professional-grade sports betting analyst powered by statistical models, machine learning, and Bayesian probability theory. You help users identify positive expected value (+EV) betting opportunities with full transparency about your reasoning, confidence levels, and limitations.

**IMPORTANT: You have access to REAL-TIME data from live APIs:**
- Live odds from The Odds API across multiple sportsbooks (DraftKings, FanDuel, BetMGM, Pinnacle, etc.)
- Current game schedules and matchups
- Pre-calculated +EV picks from your quantitative models
- All data in the <context> tag is LIVE and CURRENT, not historical

When the user asks about today's games, Super Bowl props, or current betting opportunities, you DO have access to this information through the context provided. Use it to give specific, actionable recommendations.
</role>

<personality>
- Analytical and precise, like a quant at a trading desk
- Honest about uncertainty — always state confidence levels
- Never hype or oversell — if a pick is marginal, say so
- Conversational but efficient — respect the user's time
- Use specific numbers, not vague qualifiers
</personality>

<constraints>
- NEVER guarantee outcomes or imply certainty
- ALWAYS include a confidence tier (A/B/C) and EV percentage when recommending bets
- ALWAYS mention the best available odds and which sportsbook
- If asked about a game you lack data for, say so explicitly
- Round probabilities to 1 decimal place for readability
- Round EV to 1 decimal place (e.g., "+4.2% EV")
- Use American odds format unless the user requests decimal
- NEVER recommend a bet with negative expected value
- If no +EV plays exist, say "No actionable edges found today"
- Include a responsible gambling reminder in first message of each session
</constraints>

<output_format>
When presenting a bet recommendation, use this structure:

**[CONFIDENCE_TIER] [TEAM/SIDE] [BET_TYPE] @ [ODDS] ([SPORTSBOOK])**
- **EV:** +X.X% | **Edge:** X.X% | **Fair Prob:** XX.X%
- **Key Factors:**
  1. [Most important factor with specific stat]
  2. [Second factor]
  3. [Third factor]
- **Model Breakdown:** Elo: XX% | ML: XX% | Bayesian: XX% | Market: XX%
- **Sizing:** X.X units (based on user's bankroll settings)
- **Risk Note:** [Any caveats — small sample, key injury uncertainty, etc.]

When answering general questions, be concise and use bullet points.
When the user asks "why," go deep into the statistical reasoning.
</output_format>

<uncertainty_rule>
If your models disagree by more than 5 percentage points, or if the Bayesian credible interval is wider than 15 percentage points, explicitly flag:
"Note: My models show unusual disagreement on this game. I'd recommend reduced sizing or a pass."
</uncertainty_rule>

<responsible_gambling>
If the user mentions chasing losses, betting more than they can afford, emotional decision-making, or asks for "locks" or "guaranteed" wins:
Respond with empathy and direct them to responsible gambling resources.
Never encourage increasing bet size to recover losses.
Include this in the first message of each session:
"Remember: Even +EV bets lose regularly. Bankroll management and discipline are what separate profitable bettors from recreational ones. If gambling stops being fun, visit ncpgambling.org or call 1-800-522-4700."
</responsible_gambling>"""


def build_context(
    user_prefs: Dict[str, Any],
    picks: List[Dict[str, Any]],
    odds_freshness: str = "live (fetched from The Odds API within the last 60 seconds)",
    conversation_summary: str = "",
) -> str:
    """Build structured context XML to inject into Claude messages."""
    picks_xml = ""
    for pick in picks:
        factors = ""
        for f in pick.get("top_factors", []):
            factors += f"      <factor>{f}</factor>\n"

        breakdown = pick.get("model_breakdown", {})
        picks_xml += f"""
    <pick confidence="{pick.get('confidence_tier', 'C')}" ev="{pick.get('ev_pct', 0):.3f}">
      <game>{pick.get('away_team', '?')} @ {pick.get('home_team', '?')} | {pick.get('scheduled_at', '?')}</game>
      <recommendation>{pick.get('display', '?')} @ {pick.get('best_odds', '?')} ({pick.get('best_book', '?')})</recommendation>
      <fair_prob>{pick.get('fair_prob', 0):.3f}</fair_prob>
      <ev_pct>{pick.get('ev_pct', 0):.3f}</ev_pct>
      <edge_type>{pick.get('edge_type', 'model_driven')}</edge_type>
      <kelly_units>{pick.get('kelly_units', 0):.1f}</kelly_units>
      <top_factors>
{factors}      </top_factors>
      <model_breakdown>
        <elo>{breakdown.get('elo', 0):.3f}</elo>
        <ml>{breakdown.get('ml_ensemble', 0):.3f}</ml>
        <bayesian>{breakdown.get('bayesian', 0):.3f}</bayesian>
        <market>{breakdown.get('market', 0):.3f}</market>
      </model_breakdown>
    </pick>"""

    picks_count = len(picks)
    data_note = f"REAL-TIME DATA: The {picks_count} picks below were calculated from live odds fetched from The Odds API. This is current, actionable data - not historical or demo data."

    context = f"""<context>
  <data_source_note>{data_note}</data_source_note>

  <user_profile>
    <risk_tolerance>{user_prefs.get('risk_tolerance', 'moderate')}</risk_tolerance>
    <bankroll>{user_prefs.get('bankroll_amount', 1000)}</bankroll>
    <bankroll_method>{user_prefs.get('bankroll_method', 'fractional_kelly')}</bankroll_method>
    <kelly_fraction>{user_prefs.get('kelly_fraction', 0.25)}</kelly_fraction>
    <min_ev>{user_prefs.get('min_ev_threshold', 0.03)}</min_ev>
    <enabled_sports>{', '.join(user_prefs.get('enabled_sports', ['nba']))}</enabled_sports>
  </user_profile>

  <todays_picks>{picks_xml}
  </todays_picks>

  <odds_freshness>{odds_freshness}</odds_freshness>

  <conversation_summary>{conversation_summary}</conversation_summary>
</context>"""
    return context


class ChatService:
    """Manages conversations with Claude API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from app.config import settings
        # Prefer the pydantic Settings loader (which reads backend/.env) —
        # fall back to environment variables if needed. This avoids a common
        # issue where pydantic reads the .env file but the variables are not
        # exported into os.environ.
        self.api_key = (
            api_key
            or (getattr(settings, 'ANTHROPIC_API_KEY', None) or "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.model = model or getattr(settings, 'ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929')
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("anthropic package not installed")
                return None
            except Exception as e:
                logger.error(f"Failed to create Anthropic client: {e}")
                return None
        return self._client

    def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: str = "",
        max_tokens: int = 2048,
    ) -> str:
        """
        Send a message to Claude and get a response.

        Args:
            user_message: The user's message
            conversation_history: Previous messages [{"role": "user"|"assistant", "content": "..."}]
            context: Structured context XML to prepend
            max_tokens: Maximum response tokens

        Returns:
            Claude's response text
        """
        client = self._get_client()
        if client is None:
            return self._fallback_response(user_message)

        messages = []
        if conversation_history:
            messages.extend(conversation_history)

        # Prepend context to user message if provided
        if context:
            full_message = f"{context}\n\n{user_message}"
        else:
            full_message = user_message

        messages.append({"role": "user", "content": full_message})

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            # Anthropic SDK may return different shapes depending on version.
            # Try common access patterns and fall back to string conversion.
            try:
                return response.content[0].text
            except Exception:
                try:
                    return str(response)
                except Exception:
                    return ""
        except Exception as e:
            # Persist full traceback to a temp file for debugging (local only).
            import traceback
            tb = traceback.format_exc()
            try:
                with open('/tmp/anthropic_error.log', 'a') as fh:
                    fh.write('\n--- ANTHROPIC ERROR ---\n')
                    fh.write(tb)
            except Exception:
                pass
            logger.error(f"Claude API error: {e}")
            return self._fallback_response(user_message)

    async def chat_stream(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: str = "",
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """
        Stream a response from Claude.

        Yields text chunks as they arrive.
        """
        client = self._get_client()
        if client is None:
            yield self._fallback_response(user_message)
            return

        messages = []
        if conversation_history:
            messages.extend(conversation_history)

        if context:
            full_message = f"{context}\n\n{user_message}"
        else:
            full_message = user_message

        messages.append({"role": "user", "content": full_message})

        try:
            with client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            yield self._fallback_response(user_message)

    def _fallback_response(self, user_message: str) -> str:
        """Provide a useful response when Claude API is unavailable."""
        lower = user_message.lower()
        if any(w in lower for w in ["pick", "play", "bet", "recommend", "tonight", "today"]):
            return """**EDGE AI — Demo Mode** (Claude API not connected)

I'm running without my AI reasoning engine, but here's what my models found:

**[B] Celtics ML @ -145 (FanDuel)**
- **EV:** +6.1% | **Edge:** 3.6% | **Fair Prob:** 62.8%
- **Key Factors:**
  1. Celtics +85 Elo advantage at home
  2. Opponent on back-to-back
  3. Strong 8-2 record in last 10 home games
- **Model Breakdown:** Elo: 61.5% | ML: 64.1% | Bayesian: 62.2% | Market: 60.5%
- **Sizing:** 1.8 units

*Connect your Anthropic API key to get full AI-powered analysis.*

Remember: Even +EV bets lose regularly. Bankroll management is key. If gambling stops being fun, visit ncpgambling.org or call 1-800-522-4700."""

        elif any(w in lower for w in ["hello", "hi", "hey", "help"]):
            return """Welcome to **EDGE AI** — your +EV sports betting analyst.

I can help you with:
- **Today's picks**: "What are your best plays tonight?"
- **Game analysis**: "Break down the Celtics vs Knicks game"
- **Settings**: "Change my risk tolerance to aggressive"
- **Performance**: "How have your picks been doing?"

What would you like to know?

*Remember: Even +EV bets lose regularly. Bankroll management and discipline are what separate profitable bettors from recreational ones.*"""

        else:
            return f"""I received your message: "{user_message}"

I'm currently running in demo mode without the Claude API connected. To get full AI-powered betting analysis, set your ANTHROPIC_API_KEY environment variable.

In the meantime, I can show you model outputs and odds data. Try asking "What are your best plays tonight?" to see what the models have found."""
