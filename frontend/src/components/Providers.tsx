'use client';

import { AlienProvider } from '@alien_org/react';

export function Providers({ children }: { children: React.ReactNode }) {
  return <AlienProvider>{children}</AlienProvider>;
}
