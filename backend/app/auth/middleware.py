"""Auth middleware - JWT verification via Supabase."""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from app.db.supabase_client import get_admin_client

security = HTTPBearer(auto_error=False)


async def verify_token(token: str) -> dict:
    """Verify a Supabase access token by calling Supabase auth.getUser."""
    try:
        client = get_admin_client()
        response = client.auth.get_user(token)
        if response and response.user:
            return {
                "id": response.user.id,
                "email": response.user.email or "",
            }
        raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """FastAPI dependency that extracts and verifies the user from JWT."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_data = await verify_token(credentials.credentials)
    user_data["token"] = credentials.credentials
    return user_data


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """Optional auth - returns None if no token provided."""
    if not credentials:
        return None
    try:
        user_data = await verify_token(credentials.credentials)
        user_data["token"] = credentials.credentials
        return user_data
    except HTTPException:
        return None
