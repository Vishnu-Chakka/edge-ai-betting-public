"""Wallet and balance API routes for Alien integration."""
from __future__ import annotations
from fastapi import APIRouter, Header, HTTPException
from typing import Optional
import jwt

router = APIRouter()


@router.get("/balance")
async def get_wallet_balance(authorization: Optional[str] = Header(None)):
    """
    Get wallet balance for authenticated Alien user.

    TODO: Implement actual Alien API integration
    - Decode the authToken JWT to get user info
    - Query Alien's backend API for actual wallet balance
    - Handle errors and edge cases

    For now, returns mock data for development.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    auth_token = authorization.replace("Bearer ", "")

    try:
        # Decode JWT (without verification for now - add verification in production)
        # In production, you'd verify the signature with Alien's public key
        decoded = jwt.decode(auth_token, options={"verify_signature": False})

        # TODO: Use decoded token to query Alien's API for actual balance
        # For now, return mock balance based on token presence

        # Mock balance - replace with actual API call
        return {
            "balance": 1000.0,
            "currency": "ALIEN",
            "user_id": decoded.get("sub", "unknown"),
            "last_updated": "2026-02-08T00:00:00Z"
        }

    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid auth token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch balance: {str(e)}")
