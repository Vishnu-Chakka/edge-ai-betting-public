"use client";

import { useState, useEffect } from "react";
import { Save, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { fetchPreferences, updatePreferences, type UserPreferences } from "@/lib/api";
import { useAppStore } from "@/lib/store";

const SPORTS = [
  { key: "nba", label: "NBA", emoji: "\u{1F3C0}" },
  { key: "nfl", label: "NFL", emoji: "\u{1F3C8}" },
  { key: "mlb", label: "MLB", emoji: "\u26BE" },
  { key: "nhl", label: "NHL", emoji: "\u{1F3D2}" },
  { key: "soccer", label: "Soccer", emoji: "\u26BD" },
];

const RISK_LEVELS: { value: UserPreferences["risk_tolerance"]; label: string; desc: string }[] = [
  {
    value: "conservative",
    label: "Conservative",
    desc: "Higher EV threshold, smaller bet sizes, A-tier picks only",
  },
  {
    value: "moderate",
    label: "Moderate",
    desc: "Balanced approach with A and B-tier picks",
  },
  {
    value: "aggressive",
    label: "Aggressive",
    desc: "Lower EV threshold, larger bet sizes, all tiers",
  },
];

const BANKROLL_METHODS: { value: UserPreferences["bankroll_method"]; label: string; desc: string }[] = [
  { value: "flat", label: "Flat Betting", desc: "Same amount on every bet" },
  {
    value: "fractional_kelly",
    label: "Fractional Kelly",
    desc: "Kelly criterion scaled by fraction (recommended)",
  },
  { value: "kelly", label: "Full Kelly", desc: "Maximum growth rate (volatile)" },
];

export default function SettingsPage() {
  const [prefs, setPrefs] = useState<UserPreferences | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { setPreferences: setStorePreferences } = useAppStore();

  useEffect(() => {
    loadPreferences();
  }, []);

  async function loadPreferences() {
    setLoading(true);
    try {
      const data = await fetchPreferences();
      setPrefs(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load preferences");
      setPrefs({
        enabled_sports: ["nba", "nfl", "mlb", "nhl"],
        risk_tolerance: "moderate",
        bankroll_amount: 1000,
        bankroll_method: "fractional_kelly",
        kelly_fraction: 0.25,
        min_ev_threshold: 3.0,
      });
    } finally {
      setLoading(false);
    }
  }

  async function handleSave() {
    if (!prefs) return;
    setSaving(true);
    setSaved(false);
    setError(null);

    try {
      const updated = await updatePreferences(prefs);
      setPrefs(updated);
      setStorePreferences(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save preferences");
    } finally {
      setSaving(false);
    }
  }

  function toggleSport(sport: string) {
    if (!prefs) return;
    const enabled = prefs.enabled_sports.includes(sport)
      ? prefs.enabled_sports.filter((s) => s !== sport)
      : [...prefs.enabled_sports, sport];
    setPrefs({ ...prefs, enabled_sports: enabled });
  }

  if (loading || !prefs) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-emerald-400" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-100">Settings</h1>
        <p className="mt-1 text-sm text-gray-500">
          Configure your betting preferences and risk parameters
        </p>
      </div>

      <div className="space-y-8">
        {/* Enabled Sports */}
        <section>
          <h2 className="mb-3 text-lg font-semibold text-gray-200">
            Enabled Sports
          </h2>
          <p className="mb-4 text-sm text-gray-500">
            Select which sports to include in picks and analysis.
          </p>
          <div className="flex flex-wrap gap-3">
            {SPORTS.map((sport) => {
              const isEnabled = prefs.enabled_sports.includes(sport.key);
              return (
                <button
                  key={sport.key}
                  onClick={() => toggleSport(sport.key)}
                  className={`flex items-center gap-2 rounded-xl border px-4 py-3 text-sm font-medium transition-all ${
                    isEnabled
                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                      : "border-gray-700 bg-gray-900 text-gray-500 hover:border-gray-600"
                  }`}
                >
                  <span className="text-lg">{sport.emoji}</span>
                  {sport.label}
                </button>
              );
            })}
          </div>
        </section>

        {/* Risk Tolerance */}
        <section>
          <h2 className="mb-3 text-lg font-semibold text-gray-200">
            Risk Tolerance
          </h2>
          <div className="space-y-2">
            {RISK_LEVELS.map((level) => (
              <button
                key={level.value}
                onClick={() =>
                  setPrefs({ ...prefs, risk_tolerance: level.value })
                }
                className={`w-full rounded-xl border p-4 text-left transition-all ${
                  prefs.risk_tolerance === level.value
                    ? "border-emerald-500/30 bg-emerald-500/5"
                    : "border-gray-800 bg-gray-900 hover:border-gray-700"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span
                    className={`text-sm font-semibold ${
                      prefs.risk_tolerance === level.value
                        ? "text-emerald-400"
                        : "text-gray-300"
                    }`}
                  >
                    {level.label}
                  </span>
                  <div
                    className={`h-4 w-4 rounded-full border-2 ${
                      prefs.risk_tolerance === level.value
                        ? "border-emerald-400 bg-emerald-400"
                        : "border-gray-600"
                    }`}
                  />
                </div>
                <p className="mt-1 text-xs text-gray-500">{level.desc}</p>
              </button>
            ))}
          </div>
        </section>

        {/* Bankroll Settings */}
        <section>
          <h2 className="mb-3 text-lg font-semibold text-gray-200">
            Bankroll
          </h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label
                htmlFor="bankroll"
                className="mb-1 block text-xs font-medium text-gray-400"
              >
                Bankroll Amount ($)
              </label>
              <input
                id="bankroll"
                type="number"
                min="0"
                step="100"
                value={prefs.bankroll_amount}
                onChange={(e) =>
                  setPrefs({
                    ...prefs,
                    bankroll_amount: parseFloat(e.target.value) || 0,
                  })
                }
                className="w-full rounded-lg border border-gray-700 bg-gray-900 px-4 py-2.5 text-sm text-gray-100 outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/25"
              />
            </div>
            <div>
              <label
                htmlFor="minEv"
                className="mb-1 block text-xs font-medium text-gray-400"
              >
                Minimum EV Threshold (%)
              </label>
              <input
                id="minEv"
                type="number"
                min="0"
                max="30"
                step="0.5"
                value={prefs.min_ev_threshold}
                onChange={(e) =>
                  setPrefs({
                    ...prefs,
                    min_ev_threshold: parseFloat(e.target.value) || 0,
                  })
                }
                className="w-full rounded-lg border border-gray-700 bg-gray-900 px-4 py-2.5 text-sm text-gray-100 outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/25"
              />
            </div>
          </div>
        </section>

        {/* Bankroll Method */}
        <section>
          <h2 className="mb-3 text-lg font-semibold text-gray-200">
            Sizing Method
          </h2>
          <div className="space-y-2">
            {BANKROLL_METHODS.map((method) => (
              <button
                key={method.value}
                onClick={() =>
                  setPrefs({ ...prefs, bankroll_method: method.value })
                }
                className={`w-full rounded-xl border p-4 text-left transition-all ${
                  prefs.bankroll_method === method.value
                    ? "border-emerald-500/30 bg-emerald-500/5"
                    : "border-gray-800 bg-gray-900 hover:border-gray-700"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span
                    className={`text-sm font-semibold ${
                      prefs.bankroll_method === method.value
                        ? "text-emerald-400"
                        : "text-gray-300"
                    }`}
                  >
                    {method.label}
                  </span>
                  <div
                    className={`h-4 w-4 rounded-full border-2 ${
                      prefs.bankroll_method === method.value
                        ? "border-emerald-400 bg-emerald-400"
                        : "border-gray-600"
                    }`}
                  />
                </div>
                <p className="mt-1 text-xs text-gray-500">{method.desc}</p>
              </button>
            ))}
          </div>

          {(prefs.bankroll_method === "fractional_kelly" ||
            prefs.bankroll_method === "kelly") && (
            <div className="mt-4">
              <label
                htmlFor="kellyFrac"
                className="mb-1 block text-xs font-medium text-gray-400"
              >
                Kelly Fraction (0.1 = 10% Kelly, 0.25 = Quarter Kelly)
              </label>
              <input
                id="kellyFrac"
                type="number"
                min="0.05"
                max="1.0"
                step="0.05"
                value={prefs.kelly_fraction}
                onChange={(e) =>
                  setPrefs({
                    ...prefs,
                    kelly_fraction: parseFloat(e.target.value) || 0.25,
                  })
                }
                className="w-full rounded-lg border border-gray-700 bg-gray-900 px-4 py-2.5 text-sm text-gray-100 outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/25"
              />
            </div>
          )}
        </section>

        {/* Error / Success */}
        {error && (
          <div className="flex items-center gap-3 rounded-xl border border-red-500/20 bg-red-500/10 p-4">
            <AlertCircle className="h-5 w-5 flex-shrink-0 text-red-400" />
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        {saved && (
          <div className="flex items-center gap-3 rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-4">
            <CheckCircle2 className="h-5 w-5 flex-shrink-0 text-emerald-400" />
            <p className="text-sm text-emerald-400">
              Settings saved successfully!
            </p>
          </div>
        )}

        {/* Save Button */}
        <div className="flex justify-end pb-10">
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 rounded-xl bg-emerald-500 px-8 py-3 text-sm font-semibold text-gray-950 transition-all hover:bg-emerald-400 disabled:opacity-50"
          >
            {saving ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
}
