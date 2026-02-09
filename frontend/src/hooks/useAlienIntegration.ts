'use client';

import { useAlien, useEvent, useLaunchParams } from '@alien_org/react';
import { useEffect } from 'react';

export function useAlienIntegration() {
  const { isBridgeAvailable, authToken, contractVersion } = useAlien();
  const launchParams = useLaunchParams();

  // Handle back button in Alien app
  useEvent('host.back.button:clicked', () => {
    if (typeof window !== 'undefined') {
      window.history.back();
    }
  });

  // Log Alien environment info (dev only)
  useEffect(() => {
    if (isBridgeAvailable && process.env.NODE_ENV === 'development') {
      console.log('[Alien] Running in Alien app', {
        contractVersion,
        platform: launchParams?.platform,
        hostVersion: launchParams?.hostAppVersion,
      });
    }
  }, [isBridgeAvailable, contractVersion, launchParams]);

  return {
    isAlienApp: isBridgeAvailable,
    authToken,
    platform: launchParams?.platform,
    safeAreaInsets: launchParams?.safeAreaInsets,
  };
}
