"""Auth middleware - local JWT verification using Supabase JWKS."""

import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from app.config import settings

security = HTTPBearer(auto_error=False)

_jwk_client: Optional[PyJWKClient] = None


def _get_jwk_client() -> PyJWKClient:
    global _jwk_client
    if _jwk_client is None:
        jwks_url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
        _jwk_client = PyJWKClient(jwks_url, cache_keys=True)
    return _jwk_client


async def verify_token(token: str) -> dict:
    """Verify a Supabase access token locally using JWKS public keys."""
    try:
        jwk_client = _get_jwk_client()
        signing_key = jwk_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "HS256"],
            audience="authenticated",
        )
        return {
            "id": payload["sub"],
            "email": payload.get("email", ""),
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")


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
