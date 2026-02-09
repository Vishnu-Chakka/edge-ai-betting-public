"use client";

import type { Pick } from "@/lib/api";
import {
  TrendingUp,
  Clock,
  Target,
  DollarSign,
  AlertCircle,
  Check,
} from "lucide-react";
import { useWallet } from "./WalletBalance";
import { useAuth } from "@/contexts/AuthContext";
import { useState } from "react";
import { usePayment } from "@alien_org/react";

interface BetCardProps {
  pick: Pick;
}

function getTierStyles(tier: "A" | "B" | "C") {
  switch (tier) {
    case "A":
      return {
        bg: "bg-emerald-500/20",
        text: "text-emerald-400",
        border: "border-emerald-500/30",
        borderColor: "border-emerald-500",
        label: "A - Strong",
      };
    case "B":
      return {
        bg: "bg-yellow-500/20",
        text: "text-yellow-400",
        border: "border-yellow-500/30",
        borderColor: "border-yellow-400",
        label: "B - Moderate",
      };
    case "C":
      return {
        bg: "bg-gray-500/20",
        text: "text-gray-400",
        border: "border-gray-500/30",
        borderColor: "border-gray-400",
        label: "C - Marginal",
      };
  }
}

function formatOdds(odds: number): string {
  if (odds >= 0) return `+${odds}`;
  return `${odds}`;
}

function formatTime(dateString: string): string {
  try {
    const date = new Date(dateString);
    return date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
  } catch {
    return dateString;
  }
}

/**
 * Calculate intelligent bet sizing based on available balance and EV.
 * Returns suggested bet amount that maximizes EV while respecting balance constraints.
 */
function calculateSmartBetAmount(
  recommendedAmount: number,
  availableBalance: number,
  ev: number,
  kellyFraction: number
): {
  suggestedAmount: number;
  isReduced: boolean;
  evScalingFactor: number;
  expectedProfit: number;
} {
  // If balance is sufficient, use recommended amount
  if (availableBalance >= recommendedAmount) {
    return {
      suggestedAmount: recommendedAmount,
      isReduced: false,
      evScalingFactor: 1.0,
      expectedProfit: recommendedAmount * (ev / 100),
    };
  }

  // Calculate reduced amount while maintaining Kelly principles
  // Use fractional Kelly to stay conservative when balance-constrained
  const conservativeKelly = kellyFraction * 0.75; // More conservative when limited
  const suggestedAmount = Math.min(
    availableBalance,
    availableBalance * conservativeKelly
  );

  // Round to nearest dollar for cleaner UX
  const roundedAmount = Math.max(1, Math.floor(suggestedAmount));

  return {
    suggestedAmount: roundedAmount,
    isReduced: true,
    evScalingFactor: roundedAmount / recommendedAmount,
    expectedProfit: roundedAmount * (ev / 100),
  };
}

export default function BetCard({ pick }: BetCardProps) {
  const tier = getTierStyles(pick.analysis.confidence_tier);
  const modelEntries = Object.entries(pick.analysis.model_breakdown);
  const maxModelValue = Math.max(...modelEntries.map(([, v]) => v), 1);

  const { balance, refreshBalance } = useWallet();
  const { isAlienUser, isAuthenticated } = useAuth();
  const [isPlacingBet, setIsPlacingBet] = useState(false);
  const [betPlaced, setBetPlaced] = useState(false);

  const recommendedAmount = pick.sizing.recommended_amount || 0;

  // Calculate smart bet sizing based on available balance
  const smartBet = calculateSmartBetAmount(
    recommendedAmount,
    balance,
    pick.analysis.ev_percentage,
    pick.sizing.kelly_fraction
  );

  const hasBalanceIssue = isAlienUser && balance < recommendedAmount;
  const affordableAmount = smartBet.suggestedAmount;

  const payment = usePayment({
    onPaid: async (txHash) => {
      console.log('Bet placed successfully:', txHash);
      setBetPlaced(true);
      await refreshBalance();
      setTimeout(() => setBetPlaced(false), 3000);
    },
    onCancelled: () => {
      console.log('Bet cancelled');
      setIsPlacingBet(false);
    },
    onFailed: (errorCode, error) => {
      console.error('Bet failed:', errorCode, error);
      setIsPlacingBet(false);
    },
  });

  const handlePlaceBet = async () => {
    if (!isAuthenticated || !recommendedAmount) return;

    setIsPlacingBet(true);
    try {
      // In production, create invoice on backend first
      const invoice = `bet-${Date.now()}-${Math.random().toString(36).substring(7)}`;

      await payment.pay({
        recipient: process.env.NEXT_PUBLIC_HOUSE_WALLET || 'YOUR_WALLET_ADDRESS',
        amount: Math.floor(affordableAmount * 1000000).toString(), // Convert to micro units
        token: 'ALIEN',
        network: 'alien',
        invoice,
        item: {
          title: `${pick.pick.type}: ${pick.pick.selection}`,
          iconUrl: 'https://via.placeholder.com/64',
          quantity: 1,
        },
        test: 'paid', // Remove in production
      });
    } catch (error) {
      console.error('Error placing bet:', error);
      setIsPlacingBet(false);
    }
  };

  return (
    <div className={`rounded-xl border border-gray-800 bg-gradient-to-br from-black/20 to-transparent p-5 transition-all hover:border-gray-700 hover:shadow-lg hover:shadow-emerald-500/5 border-l-4 ${tier.borderColor}`}>
      {/* Header: Sport badge + Tier badge */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="rounded-md bg-gray-800 px-2.5 py-1 text-xs font-medium text-gray-300 uppercase">
            {pick.sport}
          </span>
          <span className="text-xs text-gray-500">{pick.league}</span>
        </div>
        <span
          className={`rounded-full border px-3 py-1 text-xs font-semibold ${tier.bg} ${tier.text} ${tier.border}`}
        >
          {tier.label}
        </span>
      </div>

      {/* Teams */}
      <div className="mb-3">
        <h3 className="text-lg font-semibold text-gray-100">
          {pick.game.away_team}{" "}
          <span className="text-gray-500">@</span>{" "}
          {pick.game.home_team}
        </h3>
        <div className="mt-1 flex items-center gap-1 text-xs text-gray-500">
          <Clock className="h-3 w-3" />
          {formatTime(pick.game.scheduled_time)}
          {pick.game.venue && ` - ${pick.game.venue}`}
        </div>
      </div>

      {/* Pick details */}
      <div className="mb-4 rounded-lg bg-gray-800/50 p-3">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs text-gray-500 uppercase">{pick.pick.type}</p>
            <p className="text-sm font-semibold text-gray-100">
              {pick.pick.selection}
              {pick.pick.line !== undefined && (
                <span className="ml-1 text-gray-400">
                  ({pick.pick.line > 0 ? "+" : ""}
                  {pick.pick.line})
                </span>
              )}
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs text-gray-500">Odds</p>
            <p className="text-sm font-bold text-gray-100">
              {formatOdds(pick.pick.odds)}
            </p>
          </div>
        </div>
      </div>

      {/* Stats Row */}
      <div className="mb-4 grid grid-cols-3 gap-3">
        <div className="rounded-lg bg-gray-800/30 p-2 text-center">
          <div className="flex items-center justify-center gap-1">
            <TrendingUp className="h-3 w-3 text-emerald-400" />
            <span className="text-xs text-gray-500">EV</span>
          </div>
          <p className="text-sm font-bold text-emerald-400">
            {pick.analysis.ev_percentage.toFixed(1)}%
          </p>
        </div>
        <div className="rounded-lg bg-gray-800/30 p-2 text-center">
          <div className="flex items-center justify-center gap-1">
            <Target className="h-3 w-3 text-blue-400" />
            <span className="text-xs text-gray-500">Edge</span>
          </div>
          <p className="text-sm font-bold text-blue-400">
            {pick.analysis.edge_percentage.toFixed(1)}%
          </p>
        </div>
        <div className="rounded-lg bg-gray-800/30 p-2 text-center">
          <div className="flex items-center justify-center gap-1">
            <DollarSign className="h-3 w-3 text-purple-400" />
            <span className="text-xs text-gray-500">Fair Prob</span>
          </div>
          <p className="text-sm font-bold text-purple-400">
            {(pick.analysis.fair_probability * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Model Breakdown */}
      {modelEntries.length > 0 && (
        <div className="mb-4">
          <h4 className="mb-2 text-xs font-semibold text-gray-400 uppercase">
            Model Breakdown
          </h4>
          <div className="space-y-1.5">
            {modelEntries.map(([model, value]) => (
              <div key={model} className="flex items-center gap-2">
                <span className="w-24 truncate text-xs text-gray-500">
                  {model}
                </span>
                <div className="flex-1 overflow-hidden rounded-full bg-gray-800 h-2">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-emerald-600 to-emerald-400 transition-all"
                    style={{
                      width: `${(value / maxModelValue) * 100}%`,
                    }}
                  />
                </div>
                <span className="w-12 text-right text-xs font-medium text-gray-400">
                  {(value * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key Factors */}
      {pick.analysis.key_factors.length > 0 && (
        <div className="mb-4">
          <h4 className="mb-2 text-xs font-semibold text-gray-400 uppercase">
            Key Factors
          </h4>
          <ul className="space-y-1">
            {pick.analysis.key_factors.map((factor, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-xs text-gray-400"
              >
                <span className="mt-1 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-emerald-500" />
                {factor}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Sizing recommendation */}
      <div className="flex items-center justify-between rounded-lg border border-gray-800 bg-gradient-to-r from-black/10 to-transparent p-3">
        <div>
          <p className="text-xs text-gray-500">Recommended Size</p>
          <p className="text-sm font-semibold text-gray-200">
            {pick.sizing.recommended_units.toFixed(2)} units
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-gray-500">Kelly Fraction</p>
          <p className="text-sm font-medium text-gray-300">
            {(pick.sizing.kelly_fraction * 100).toFixed(1)}%
          </p>
        </div>
        {pick.sizing.recommended_amount !== undefined && (
          <div className="text-right">
            <p className="text-xs text-gray-500">Amount</p>
            <p className="text-sm font-semibold text-emerald-400">
              ${pick.sizing.recommended_amount.toFixed(0)}
            </p>
          </div>
        )}
      </div>

      {/* Balance-aware betting section (Alien users only) */}
      {isAlienUser && isAuthenticated && recommendedAmount > 0 && (
        <div className="mt-3 space-y-2">
          {/* Smart bet sizing info */}
          <div className={`rounded-lg border p-3 ${
            hasBalanceIssue
              ? 'border-yellow-500/30 bg-yellow-500/10'
              : 'border-emerald-500/30 bg-emerald-500/10'
          }`}>
            <div className="flex items-start gap-2">
              {hasBalanceIssue ? (
                <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-yellow-400" />
              ) : (
                <Check className="mt-0.5 h-4 w-4 flex-shrink-0 text-emerald-400" />
              )}
              <div className="flex-1 space-y-1.5">
                <p className={`text-xs font-medium ${
                  hasBalanceIssue ? 'text-yellow-400' : 'text-emerald-400'
                }`}>
                  {hasBalanceIssue ? 'Smart Bet Adjustment' : 'Optimal Bet Size'}
                </p>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <p className={hasBalanceIssue ? 'text-yellow-300/70' : 'text-emerald-300/70'}>
                      Your Balance
                    </p>
                    <p className={`font-semibold ${hasBalanceIssue ? 'text-yellow-300' : 'text-emerald-300'}`}>
                      ${balance.toFixed(0)} ALIEN
                    </p>
                  </div>
                  <div>
                    <p className={hasBalanceIssue ? 'text-yellow-300/70' : 'text-emerald-300/70'}>
                      Suggested Bet
                    </p>
                    <p className={`font-semibold ${hasBalanceIssue ? 'text-yellow-300' : 'text-emerald-300'}`}>
                      ${smartBet.suggestedAmount.toFixed(0)}
                      {smartBet.isReduced && (
                        <span className="ml-1 text-[10px] opacity-70">
                          ({(smartBet.evScalingFactor * 100).toFixed(0)}% of rec.)
                        </span>
                      )}
                    </p>
                  </div>
                </div>

                <div className={`mt-2 pt-2 border-t ${
                  hasBalanceIssue ? 'border-yellow-500/20' : 'border-emerald-500/20'
                }`}>
                  <p className={hasBalanceIssue ? 'text-yellow-300/70' : 'text-emerald-300/70'}>
                    Expected Value (EV)
                  </p>
                  <p className={`font-bold ${hasBalanceIssue ? 'text-yellow-300' : 'text-emerald-300'}`}>
                    +${smartBet.expectedProfit.toFixed(2)}
                    <span className="ml-1 text-xs font-normal opacity-70">
                      ({pick.analysis.ev_percentage.toFixed(1)}% EV)
                    </span>
                  </p>
                </div>

                {smartBet.isReduced && (
                  <p className="text-[10px] text-yellow-300/60 mt-1">
                    💡 Bet size reduced to maintain Kelly criterion with your available balance
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Place bet button */}
          <button
            onClick={handlePlaceBet}
            disabled={isPlacingBet || betPlaced || balance < 1}
            className={`w-full rounded-lg px-4 py-3 text-sm font-semibold transition-all ${
              betPlaced
                ? "bg-emerald-500/20 text-emerald-400 cursor-default"
                : balance < 1
                ? "bg-gray-700 text-gray-500 cursor-not-allowed"
                : hasBalanceIssue
                ? "bg-yellow-500/20 text-yellow-300 hover:bg-yellow-500/30 border border-yellow-500/30"
                : "bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30 border border-emerald-500/30"
            }`}
          >
            {betPlaced ? (
              <span className="flex items-center justify-center gap-2">
                <Check className="h-4 w-4" />
                Bet Placed Successfully
              </span>
            ) : isPlacingBet ? (
              "Processing..."
            ) : balance < 1 ? (
              "Insufficient Balance - Add Funds"
            ) : (
              `Place Bet ($${smartBet.suggestedAmount.toFixed(0)}) • EV: +$${smartBet.expectedProfit.toFixed(2)}`
            )}
          </button>

          <p className="text-center text-[10px] text-gray-600">
            Powered by Alien Wallet • {smartBet.isReduced ? 'Optimized' : 'Recommended'} Kelly sizing
          </p>
        </div>
      )}
    </div>
  );
}
