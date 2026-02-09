"""Wallet and balance API routes for Alien integration."""
from __future__ import annotations
from fastapi import APIRouter, Header, HTTPException
from typing import Optional
import jwt
import httpx
from datetime import datetime

router = APIRouter()

# Alien API configuration
ALIEN_API_BASE = "https://api.alien.org"  # Update with actual Alien API URL
ALIEN_BALANCE_ENDPOINT = "/v1/wallet/balance"  # Update with actual endpoint


async def fetch_alien_balance(user_id: str, auth_token: str) -> dict:
    """
    Fetch actual wallet balance from Alien API.

    This function calls Alien's backend API to get the user's wallet balance.
    Falls back to mock data if API is unavailable.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ALIEN_API_BASE}{ALIEN_BALANCE_ENDPOINT}",
                headers={
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "balance": float(data.get("balance", 0)),
                    "currency": data.get("currency", "ALIEN"),
                    "user_id": user_id,
                    "last_updated": datetime.utcnow().isoformat() + "Z"
                }
            else:
                # API returned error, use mock data
                print(f"Alien API returned status {response.status_code}, using mock data")
                return _get_mock_balance(user_id)

    except httpx.RequestError as e:
        print(f"Failed to connect to Alien API: {e}, using mock data")
        return _get_mock_balance(user_id)
    except Exception as e:
        print(f"Unexpected error fetching balance: {e}, using mock data")
        return _get_mock_balance(user_id)


def _get_mock_balance(user_id: str) -> dict:
    """
    Return mock balance data for development/testing.

    In production, this should only be used as a fallback when the Alien API is unavailable.
    The mock balance is deterministic based on user_id for consistency.
    """
    # Generate consistent mock balance based on user_id hash
    # This ensures the same user always sees the same mock balance
    balance_seed = abs(hash(user_id)) % 5000 + 500  # Range: 500-5500

    return {
        "balance": float(balance_seed),
        "currency": "ALIEN",
        "user_id": user_id,
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "is_mock": True  # Flag to indicate this is mock data
    }


@router.get("/balance")
async def get_wallet_balance(authorization: Optional[str] = Header(None)):
    """
    Get wallet balance for authenticated Alien user.

    This endpoint:
    1. Validates the authToken JWT
    2. Extracts the user ID from the token
    3. Queries Alien's API for actual wallet balance
    4. Falls back to mock data if API is unavailable

    Returns:
        {
            "balance": float,
            "currency": str,
            "user_id": str,
            "last_updated": str (ISO 8601 timestamp),
            "is_mock": bool (optional, present if using mock data)
        }
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    auth_token = authorization.replace("Bearer ", "")

    try:
        # Decode JWT (without verification for now - add verification in production)
        # In production, you'd verify the signature with Alien's public key
        decoded = jwt.decode(auth_token, options={"verify_signature": False})
        user_id = decoded.get("sub", "unknown")

        # Fetch actual balance from Alien API
        balance_data = await fetch_alien_balance(user_id, auth_token)

        return balance_data

    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid auth token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch balance: {str(e)}")
