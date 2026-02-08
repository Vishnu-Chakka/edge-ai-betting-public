import { create } from "zustand";
import type { Pick, UserPreferences } from "./api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface AppState {
  // Chat state
  messages: ChatMessage[];
  conversationId: string | null;
  isLoading: boolean;

  // Picks state
  picks: Pick[];

  // Preferences state
  preferences: UserPreferences;

  // Actions
  addMessage: (message: Omit<ChatMessage, "id" | "timestamp">) => void;
  setPicks: (picks: Pick[]) => void;
  setPreferences: (prefs: UserPreferences) => void;
  setLoading: (loading: boolean) => void;
  setConversationId: (id: string | null) => void;
  clearMessages: () => void;
}

const defaultPreferences: UserPreferences = {
  enabled_sports: ["NFL", "NBA", "MLB", "NHL"],
  risk_tolerance: "moderate",
  bankroll_amount: 1000,
  bankroll_method: "fractional_kelly",
  kelly_fraction: 0.25,
  min_ev_threshold: 3.0,
};

export const useAppStore = create<AppState>((set) => ({
  messages: [],
  conversationId: null,
  isLoading: false,
  picks: [],
  preferences: defaultPreferences,

  addMessage: (message) =>
    set((state) => ({
      messages: [
        ...state.messages,
        {
          ...message,
          id: crypto.randomUUID(),
          timestamp: new Date(),
        },
      ],
    })),

  setPicks: (picks) => set({ picks }),

  setPreferences: (preferences) => set({ preferences }),

  setLoading: (isLoading) => set({ isLoading }),

  setConversationId: (conversationId) => set({ conversationId }),

  clearMessages: () => set({ messages: [], conversationId: null }),
}));
