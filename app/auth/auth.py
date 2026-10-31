import secrets

from fastapi import HTTPException, Request

from app.config import config
from app.utils.logging_config import auth_logger

# Load invite codes from configuration and environment variable
# INVITE_CODES is a dictionary where keys are invite codes and values contain user info and active status
INVITE_CODES = config.invite_codes
auth_logger.info(f"Successfully loaded {len(INVITE_CODES)} total invite codes")

# Active sessions mapping session_token to invite_code to track authenticated users
active_sessions: dict[str, str] = {}  # session_token -> invite_code


def is_auth_enabled() -> bool:
    """Check if authentication is enabled based on invite codes."""
    return len(INVITE_CODES) > 0


def generate_session_token() -> str:
    """
    Generate a secure, random session token for user authentication sessions.

    Returns:
        str: A URL-safe, 32-byte random session token.
    """
    token = secrets.token_urlsafe(32)
    auth_logger.debug("Generated new session token.")
    return token


def validate_invite_code(code: str) -> bool:
    """
    Check if the given invite code exists and is currently active.

    Args:
        code (str): The invite code to validate.

    Returns:
        bool: True if the invite code is valid and active, False otherwise.
    """
    return code in INVITE_CODES and INVITE_CODES[code]["active"]


def authenticate_with_code(code: str) -> str | None:
    """
    Authenticate a user by their invite code and create a new session token upon success.

    Args:
        code (str): The invite code provided by the user.

    Returns:
        Optional[str]: A new session token if authentication is successful, None otherwise.
    """
    if validate_invite_code(code):
        session_token = generate_session_token()
        active_sessions[session_token] = code
        auth_logger.info("Successful authentication for invite code")
        return session_token
    auth_logger.warning("Failed authentication attempt with invalid invite code")
    return None


def get_current_user(request: Request) -> str | None:
    """
    Retrieve the current user's invite code from the session token cookie in the request.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        Optional[str]: The invite code associated with the current session, or None if no valid session exists.
    """
    session_token = request.cookies.get("session_token")
    if session_token and session_token in active_sessions:
        user_code = active_sessions[session_token]
        auth_logger.debug("Valid session found for request")
        return user_code
    auth_logger.debug("No valid session found in request")
    return None


def require_auth(request: Request) -> str:
    """
    Dependency function to enforce authentication on protected routes.
    If no invite codes are configured, returns a default anonymous user.

    Args:
        request (Request): The incoming FastAPI request object.

    Raises:
        HTTPException: Raises 401 Unauthorized if auth is enabled but no valid user session is found.

    Returns:
        str: The invite code of the authenticated user, or "anonymous" if auth is disabled.
    """
    # If no invite codes configured, bypass authentication
    if not is_auth_enabled():
        auth_logger.debug("Authentication bypassed - no invite codes configured")
        return "anonymous"

    user_code = get_current_user(request)
    if not user_code:
        auth_logger.warning("Authentication required but no valid session found")
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_code


def get_user_info(invite_code: str) -> dict:
    """
    Retrieve user information associated with a given invite code.

    Args:
        invite_code (str): The invite code of the user.

    Returns:
        Dict: A dictionary containing user information, or default anonymous info if not found.
    """
    if invite_code == "anonymous":
        return {
            "company": "Anonymous User",
            "recruiter": "Public Access",
            "active": True,
        }

    return config.invite_codes.get(invite_code, INVITE_CODES.get(invite_code, {}))


def logout_user(session_token: str) -> bool:
    """
    Log out the user by deleting their session token from active sessions.

    Args:
        session_token (str): The session token to invalidate.

    Returns:
        bool: True if logout was successful, False if the session token was invalid.
    """
    if session_token in active_sessions:
        del active_sessions[session_token]
        auth_logger.info("User logged out successfully")
        return True
    auth_logger.warning("Logout attempt with invalid session token")
    return False
