'use client';

import { useAuth } from '@/contexts/AuthContext';
import { Wallet, Loader2, RefreshCw } from 'lucide-react';
import { useState, useEffect, useCallback, createContext, useContext } from 'react';

interface WalletBalanceData {
  balance: number;
  currency: string;
}

interface WalletContextType {
  balance: number;
  currency: string;
  refreshBalance: () => Promise<void>;
  isLoading: boolean;
}

const WalletContext = createContext<WalletContextType>({
  balance: 0,
  currency: 'ALIEN',
  refreshBalance: async () => {},
  isLoading: false,
});

export function useWallet() {
  return useContext(WalletContext);
}

export function WalletProvider({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, authToken } = useAuth();
  const [balance, setBalance] = useState(0);
  const [currency, setCurrency] = useState('ALIEN');
  const [isLoading, setIsLoading] = useState(false);

  const refreshBalance = useCallback(async () => {
    if (!isAuthenticated || !authToken) {
      setBalance(0);
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/wallet/balance`, {
        headers: {
          'Authorization': `Bearer ${authToken}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setBalance(data.balance);
        setCurrency(data.currency);
      } else {
        // Mock balance for demo
        setBalance(1000);
        setCurrency('ALIEN');
      }
    } catch (error) {
      console.error('Failed to fetch balance:', error);
      // Mock balance for demo
      setBalance(1000);
      setCurrency('ALIEN');
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated, authToken]);

  useEffect(() => {
    refreshBalance();
  }, [refreshBalance]);

  const value = {
    balance,
    currency,
    refreshBalance,
    isLoading,
  };

  return <WalletContext.Provider value={value}>{children}</WalletContext.Provider>;
}

export function WalletBalance() {
  const { isAlienUser } = useAuth();
  const { balance, currency, refreshBalance, isLoading } = useWallet();
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refreshBalance();
    setIsRefreshing(false);
  };

  // Only show for Alien users
  if (!isAlienUser) return null;

  return (
    <div className="flex items-center gap-2 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-1.5 transition-all hover:border-violet-500/50">
      <Wallet className="h-4 w-4 text-violet-400" />
      <div className="flex flex-col">
        <span className="text-[10px] text-violet-300/70 uppercase tracking-wide">Balance</span>
        {isLoading ? (
          <div className="flex items-center gap-1">
            <Loader2 className="h-3 w-3 animate-spin text-violet-400" />
            <span className="text-xs text-violet-400">Loading...</span>
          </div>
        ) : (
          <span className="text-sm font-bold text-violet-300">
            {balance.toLocaleString()} {currency}
          </span>
        )}
      </div>
      <button
        onClick={handleRefresh}
        disabled={isRefreshing}
        className="ml-1 rounded p-1 transition-colors hover:bg-violet-500/20 disabled:opacity-50"
        title="Refresh balance"
      >
        <RefreshCw className={`h-3 w-3 text-violet-400 ${isRefreshing ? 'animate-spin' : ''}`} />
      </button>
    </div>
  );
}
