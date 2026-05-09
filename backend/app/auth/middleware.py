"""Auth middleware - JWT verification and per-request Supabase client."""

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from typing import Optional

from app.config import settings

security = HTTPBearer(auto_error=False)


def verify_jwt(token: str) -> dict:
    """Verify a Supabase JWT and return its payload."""
    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return payload
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """FastAPI dependency that extracts and verifies the user from JWT."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = verify_jwt(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token: no user ID")
    return {"id": user_id, "email": payload.get("email", ""), "token": credentials.credentials}


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """Optional auth - returns None if no token provided."""
    if not credentials:
        return None
    try:
        payload = verify_jwt(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            return None
        return {"id": user_id, "email": payload.get("email", ""), "token": credentials.credentials}
    except HTTPException:
        return None
