"""Supabase client factory - dual client strategy."""

from supabase import create_client, Client
from app.config import settings


def get_admin_client() -> Client:
    """Service-key client for internal writes (bypasses RLS)."""
    return create_client(settings.supabase_url, settings.supabase_service_key)


def get_user_client(user_jwt: str) -> Client:
    """Per-request client using user's JWT (RLS active)."""
    client = create_client(settings.supabase_url, settings.supabase_anon_key)
    client.auth.set_session(user_jwt, "")
    client.postgrest.auth(user_jwt)
    return client
