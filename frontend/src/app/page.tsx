import Link from "next/link";
import {
  MessageSquare,
  Zap,
  TrendingUp,
  Shield,
  Brain,
} from "lucide-react";

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* Hero Section (big, spacious) */}
      <section className="relative flex items-center justify-center px-6 py-28 text-center sm:py-36">
        <div className="absolute inset-0 -z-10 bg-gradient-to-b from-transparent via-black/20 to-black/50" />
        <div className="relative z-10 max-w-5xl">
          <div className="mb-6 text-center">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/6 bg-white/3 px-4 py-1.5 text-sm text-white/80">
              <Zap className="h-4 w-4 text-violet-300" />
              AI-POWERED PICKS AND INSIGHTS
            </div>
          </div>

          <h1 className="mb-6 text-center text-6xl font-extrabold leading-tight tracking-tight text-white sm:text-[96px] hero-giant">
            Smarter bets,
            <br />
            higher returns.
          </h1>

          <p className="mx-auto mb-10 max-w-3xl text-lg text-white/70">
            Get instant insights, deep analysis, and effective risk management
            powered by cutting-edge algorithms and AI models, helping you make
            confident betting decisions in seconds.
          </p>

          <div className="flex items-center justify-center gap-4">
            <Link href="/chat" className="btn-primary inline-flex items-center gap-3">
              <MessageSquare className="h-5 w-5 opacity-90" />
              START CHATTING NOW
            </Link>
          </div>

        </div>
      </section>

      {/* Features Section */}
      <section className="border-t border-white/6 px-4 py-20">
        <div className="mx-auto max-w-6xl">
          <h2 className="mb-12 text-center text-2xl font-bold text-white/90 sm:text-3xl">
            How It Works
          </h2>
          <div className="grid gap-8 md:grid-cols-3">
            <FeatureCard
              icon={Brain}
              title="Multi-Model AI Consensus"
              description="Combines predictions from multiple AI models to generate a consensus probability, reducing individual model bias and improving accuracy."
            />
            <FeatureCard
              icon={TrendingUp}
              title="Expected Value Analysis"
              description="Compares AI-derived fair odds against market lines to identify +EV opportunities with quantified edge percentages and confidence tiers."
            />
            <FeatureCard
              icon={Shield}
              title="Kelly Criterion Sizing"
              description="Calculates optimal bet sizing using fractional Kelly criterion based on your bankroll, risk tolerance, and the calculated edge on each bet."
            />
          </div>
        </div>
      </section>

      {/* Responsible Gambling Notice */}
      <section className="border-t border-white/6 px-4 py-10">
        <div className="mx-auto max-w-3xl text-center">
          <p className="text-xs text-white/60">
            <span className="font-semibold text-white/80">Responsible Gambling Notice:</span>{" "}
            Sports betting involves risk. Past performance does not guarantee
            future results. AI predictions are probabilistic estimates, not
            certainties. Never bet more than you can afford to lose. If you or
            someone you know has a gambling problem, call 1-800-GAMBLER. Must be
            21+ and located in a jurisdiction where sports betting is legal.
          </p>
        </div>
      </section>
    </div>
  );
}

function FeatureCard({
  icon: Icon,
  title,
  description,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-6 transition-colors hover:border-gray-700">
      <div className="mb-4 inline-flex rounded-lg bg-emerald-500/10 p-3">
        <Icon className="h-6 w-6 text-emerald-400" />
      </div>
      <h3 className="mb-2 text-lg font-semibold text-gray-200">{title}</h3>
      <p className="text-sm leading-relaxed text-gray-500">{description}</p>
    </div>
  );
}
