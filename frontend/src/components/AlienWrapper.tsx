'use client';

import { useAlien, useEvent } from '@alien_org/react';
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export function AlienWrapper({ children }: { children: React.ReactNode }) {
  const { isBridgeAvailable, ready } = useAlien();
  const router = useRouter();

  // Handle back button
  useEvent('host.back.button:clicked', () => {
    router.back();
  });

  // Signal ready when mounted
  useEffect(() => {
    if (isBridgeAvailable) {
      console.log('[Alien] Bridge available, signaling ready');
      ready();
    }
  }, [isBridgeAvailable, ready]);

  return <>{children}</>;
}
