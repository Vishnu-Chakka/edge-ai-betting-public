'use client';

import { createContext, useContext, ReactNode } from 'react';
import { useAlien } from '@alien_org/react';

interface AuthContextType {
  isAuthenticated: boolean;
  authToken: string | undefined;
  isAlienUser: boolean;
}

const AuthContext = createContext<AuthContextType>({
  isAuthenticated: false,
  authToken: undefined,
  isAlienUser: false,
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const { authToken, isBridgeAvailable } = useAlien();

  const value = {
    isAuthenticated: !!authToken,
    authToken,
    isAlienUser: isBridgeAvailable,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}
