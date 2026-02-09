'use client';

import { AlienProvider } from '@alien_org/react';
import { AlienWrapper } from './AlienWrapper';
import { AuthProvider } from '@/contexts/AuthContext';
import { WalletProvider } from './WalletBalance';

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <AlienProvider autoReady={false} interceptLinks={true}>
      <AlienWrapper>
        <AuthProvider>
          <WalletProvider>{children}</WalletProvider>
        </AuthProvider>
      </AlienWrapper>
    </AlienProvider>
  );
}
