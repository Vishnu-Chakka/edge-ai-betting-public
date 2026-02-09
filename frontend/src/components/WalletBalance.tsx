'use client';

import { useAuth } from '@/contexts/AuthContext';
import { Wallet, Loader2 } from 'lucide-react';
import { useState, useEffect } from 'react';

interface WalletBalanceData {
  balance: number;
  currency: string;
}

export function WalletBalance() {
  const { isAuthenticated, isAlienUser, authToken } = useAuth();
  const [balance, setBalance] = useState<WalletBalanceData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isAuthenticated || !authToken) {
      setBalance(null);
      return;
    }

    // Fetch wallet balance from backend
    const fetchBalance = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/wallet/balance`, {
          headers: {
            'Authorization': `Bearer ${authToken}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          setBalance(data);
        } else {
          // If endpoint doesn't exist yet, show placeholder
          console.log('Wallet balance endpoint not implemented yet');
          // Show mock balance for demo (remove in production)
          setBalance({ balance: 1000, currency: 'ALIEN' });
        }
      } catch (error) {
        console.error('Failed to fetch balance:', error);
        // Show mock balance for demo (remove in production)
        setBalance({ balance: 1000, currency: 'ALIEN' });
      } finally {
        setLoading(false);
      }
    };

    fetchBalance();
  }, [isAuthenticated, authToken]);

  // Only show for Alien users
  if (!isAlienUser) return null;

  if (!isAuthenticated) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-800/50 px-3 py-1.5">
        <Wallet className="h-4 w-4 text-gray-400" />
        <span className="text-xs font-medium text-gray-400">Connect Wallet</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-1.5">
      <Wallet className="h-4 w-4 text-violet-400" />
      <div className="flex flex-col">
        <span className="text-[10px] text-violet-300/70">Available Balance</span>
        {loading ? (
          <div className="flex items-center gap-1">
            <Loader2 className="h-3 w-3 animate-spin text-violet-400" />
            <span className="text-xs text-violet-400">Loading...</span>
          </div>
        ) : balance ? (
          <span className="text-sm font-bold text-violet-300">
            {balance.balance.toLocaleString()} {balance.currency}
          </span>
        ) : (
          <span className="text-xs text-violet-400">--</span>
        )}
      </div>
    </div>
  );
}
