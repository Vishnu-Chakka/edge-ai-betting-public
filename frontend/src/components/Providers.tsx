'use client';

import { AlienProvider } from '@alien_org/react';
import { AlienWrapper } from './AlienWrapper';

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <AlienProvider autoReady={false} interceptLinks={true}>
      <AlienWrapper>{children}</AlienWrapper>
    </AlienProvider>
  );
}
