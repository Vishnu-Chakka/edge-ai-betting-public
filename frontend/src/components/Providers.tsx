'use client';

import { AlienProvider } from '@alien_org/react';
import { AlienWrapper } from './AlienWrapper';
import { AuthProvider } from '@/contexts/AuthContext';

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <AlienProvider autoReady={false} interceptLinks={true}>
      <AlienWrapper>
        <AuthProvider>{children}</AuthProvider>
      </AlienWrapper>
    </AlienProvider>
  );
}
