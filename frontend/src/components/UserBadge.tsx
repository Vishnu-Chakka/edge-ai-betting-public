'use client';

import { useAuth } from '@/contexts/AuthContext';

export function UserBadge() {
  const { isAuthenticated, isAlienUser } = useAuth();

  if (!isAlienUser) return null;

  return (
    <div className="flex items-center gap-2">
      {isAuthenticated ? (
        <div className="flex items-center gap-2 rounded-full bg-emerald-500/10 px-3 py-1.5 text-sm text-emerald-400">
          <div className="h-2 w-2 rounded-full bg-emerald-400" />
          Alien User
        </div>
      ) : (
        <div className="rounded-full bg-white/5 px-3 py-1.5 text-sm text-white/50">
          Not Authenticated
        </div>
      )}
    </div>
  );
}
