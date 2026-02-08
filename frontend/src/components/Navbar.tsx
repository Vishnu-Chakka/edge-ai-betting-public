"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { MessageSquare, Settings, Zap } from "lucide-react";

const navItems = [
  { href: "/", label: "EDGE AI", icon: Zap, isLogo: true },
  { href: "/chat", label: "Chat", icon: MessageSquare, isLogo: false },
  { href: "/settings", label: "Settings", icon: Settings, isLogo: false },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-50 glass border-b border-white/5">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-4">
          {navItems
            .filter((item) => item.isLogo)
            .map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="flex items-center gap-3 text-xl font-extrabold text-white/95 transition-colors"
              >
                <item.icon className="h-6 w-6 text-violet-300" />
                <span className="bg-clip-text text-transparent bg-gradient-to-r from-violet-300 to-violet-500">
                  {item.label}
                </span>
              </Link>
            ))}
        </div>

        <div className="flex items-center gap-3">
          {navItems
            .filter((item) => !item.isLogo)
            .map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`hidden md:inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-white/6 text-white"
                      : "text-white/70 hover:bg-white/5 hover:text-white"
                  }`}
                >
                  <item.icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}

          <Link href="/signin" className="text-sm text-white/70 hover:text-white">
            Sign In
          </Link>

          <Link href="/signup" className="btn-primary hidden sm:inline-flex">
            Get Instant Access
          </Link>
        </div>
      </div>
    </nav>
  );
}
