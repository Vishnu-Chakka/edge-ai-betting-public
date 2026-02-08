"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Loader2, Trash2, Bot, User } from "lucide-react";
import { useAppStore } from "@/lib/store";
import { sendChatMessage } from "@/lib/api";

export default function ChatPage() {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const {
    messages,
    conversationId,
    isLoading,
    addMessage,
    setLoading,
    setConversationId,
    clearMessages,
  } = useAppStore();

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    if (messages.length === 0) {
      addMessage({
        role: "assistant",
        content: `Welcome to **EDGE AI** — your +EV sports betting analyst.

I can help you with:
- **Today's picks**: "What are your best plays tonight?"
- **Game analysis**: "Break down the Celtics vs Knicks game"
- **Settings**: "Change my risk tolerance to aggressive"
- **Performance**: "How have your picks been doing?"

What would you like to know?

*Remember: Even +EV bets lose regularly. Bankroll management and discipline are what separate profitable bettors from recreational ones.*`,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    addMessage({ role: "user", content: trimmed });
    setInput("");
    setLoading(true);

    try {
      const response = await sendChatMessage(trimmed, conversationId);
      addMessage({ role: "assistant", content: response.response });
      if (response.conversation_id) {
        setConversationId(response.conversation_id);
      }
    } catch (error) {
      addMessage({
        role: "assistant",
        content: `Sorry, I encountered an error processing your request. Please try again.\n\n*Error: ${
          error instanceof Error ? error.message : "Unknown error"
        }*`,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex h-[calc(100vh-3.5rem)] flex-col">
      {/* Chat Header */}
      <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-500/10">
            <Bot className="h-5 w-5 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-gray-100">EDGE AI</h1>
            <p className="text-xs text-gray-500">+EV Sports Betting Analyst</p>
          </div>
        </div>
        <button
          onClick={clearMessages}
          className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs text-gray-500 transition-colors hover:bg-gray-800 hover:text-gray-300"
          title="New conversation"
        >
          <Trash2 className="h-3.5 w-3.5" />
          New Chat
        </button>
      </div>

      {/* Messages */}
      <div className="chat-scroll flex-1 overflow-y-auto px-4 py-4 sm:px-6">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.role === "assistant" && (
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-emerald-500/10">
                  <Bot className="h-4 w-4 text-emerald-400" />
                </div>
              )}
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-emerald-600 text-white"
                    : "bg-gray-800 text-gray-200"
                }`}
              >
                <MessageContent content={msg.content} />
              </div>
              {msg.role === "user" && (
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-gray-700">
                  <User className="h-4 w-4 text-gray-300" />
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3">
              <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-emerald-500/10">
                <Bot className="h-4 w-4 text-emerald-400" />
              </div>
              <div className="rounded-2xl bg-gray-800 px-4 py-3">
                <div className="flex gap-1.5">
                  <span className="loading-dot" />
                  <span className="loading-dot" />
                  <span className="loading-dot" />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-gray-800 px-4 py-3 sm:px-6">
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex max-w-3xl items-end gap-3"
        >
          <div className="flex-1 overflow-hidden rounded-xl border border-gray-700 bg-gray-900 focus-within:border-emerald-500/50 focus-within:ring-1 focus-within:ring-emerald-500/25">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about today's best plays, game analysis, or betting strategy..."
              className="block w-full resize-none bg-transparent px-4 py-3 text-sm text-gray-100 placeholder-gray-500 outline-none"
              rows={1}
              style={{
                minHeight: "44px",
                maxHeight: "120px",
                height: "auto",
              }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = "auto";
                target.style.height = `${Math.min(target.scrollHeight, 120)}px`;
              }}
              disabled={isLoading}
            />
          </div>
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-xl bg-emerald-500 text-gray-950 transition-all hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-emerald-500"
          >
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </button>
        </form>
        <p className="mx-auto mt-2 max-w-3xl text-center text-[10px] text-gray-600">
          AI predictions are probabilistic. Never bet more than you can afford
          to lose.
        </p>
      </div>
    </div>
  );
}

function MessageContent({ content }: { content: string }) {
  const lines = content.split("\n");

  return (
    <div className="space-y-1">
      {lines.map((line, i) => {
        if (line.trim() === "") return <br key={i} />;

        if (line.trim().startsWith("- ")) {
          return (
            <div key={i} className="flex gap-2 pl-1">
              <span className="mt-2 block h-1 w-1 flex-shrink-0 rounded-full bg-current opacity-50" />
              <span
                dangerouslySetInnerHTML={{
                  __html: formatInline(line.trim().slice(2)),
                }}
              />
            </div>
          );
        }

        return (
          <p
            key={i}
            dangerouslySetInnerHTML={{ __html: formatInline(line) }}
          />
        );
      })}
    </div>
  );
}

function formatInline(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold">$1</strong>')
    .replace(/\*(.+?)\*/g, '<em class="italic opacity-80">$1</em>');
}
