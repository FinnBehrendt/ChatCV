import time

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response

from app.auth.auth import (
    authenticate_with_code,
    get_current_user,
    get_user_info,
    is_auth_enabled,
    logout_user,
    require_auth,
)
from app.modules.job_matching import (
    analyze_job_match,
    process_job_description,
)
from app.modules.rag_pipeline import get_chat_completion
from app.modules.summary_pipeline import get_auto_summary
from app.utils.analytics import AdvancedAnalytics, log_login_event
from app.utils.logging_config import api_logger

router = APIRouter()
advanced_analytics = AdvancedAnalytics()


@router.post("/auth/login")
async def login(
    request: Request, response: Response, invite_code: str = Body(..., embed=True)
):
    """
    Authenticate user with an invite code.

    Sets a secure session cookie upon successful authentication.
    Logs login events and starts a chat session for analytics.
    """
    session_token = authenticate_with_code(invite_code)
    user_info = get_user_info(invite_code)

    if not session_token:
        log_login_event(invite_code, user_info.get("company", "Unknown"), False)
        api_logger.warning(f"Login failed for invite code: {invite_code}")
        raise HTTPException(status_code=401, detail="Invalid invite code")

    log_login_event(invite_code, user_info.get("company", "Unknown"), True)
    api_logger.info(
        f"Successful login for company: {user_info.get('company', 'Unknown')}, invite code: {invite_code}"
    )

    # Start chat session in LangSmith
    advanced_analytics.start_chat_session(
        session_token, invite_code, user_info.get("company", "Unknown")
    )

    # Set secure cookie using config settings
    from app.config import config

    # For local development (localhost/127.0.0.1), force secure_cookies to False
    host = request.headers.get("host", "").lower()
    is_local = "localhost" in host or "127.0.0.1" in host
    secure_cookies = config.security.secure_cookies and not is_local

    api_logger.debug(
        f"Setting session cookie (secure={secure_cookies}, host={host}) for invite code: {invite_code}"
    )

    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=secure_cookies,
        samesite="lax",
        max_age=config.security.session_timeout_hours * 60 * 60,
    )

    return {
        "success": True,
        "message": "Authentication successful",
        "user": {"company": user_info.get("company", "Unknown"), "code": invite_code},
    }


@router.post("/auth/logout")
async def logout(request: Request, response: Response):
    """
    Log out the current user by deleting the session cookie.

    Ends the chat session in LangSmith and logs the logout event.
    """
    session_token = request.cookies.get("session_token")
    if session_token:
        # End chat session in LangSmith
        advanced_analytics.end_chat_session(session_token)
        logout_user(session_token)
        api_logger.info(f"User logged out with session token: {session_token}")
    else:
        api_logger.info("Logout attempted with no active session token found.")

    response.delete_cookie("session_token")
    return {"success": True, "message": "Logged out successfully"}


@router.get("/auth/status")
async def auth_status(request: Request):
    """
    Check the authentication status of the current user.
    Returns user information if authenticated, or anonymous status if auth is disabled.
    """
    # If authentication is disabled, return anonymous status
    if not is_auth_enabled():
        api_logger.debug("Auth status: authentication disabled")
        return {
            "authenticated": False,
            "auth_enabled": False,
            "user": {"company": "Anonymous User", "code": "anonymous"},
        }

    user_code = get_current_user(request)
    if user_code:
        user_info = get_user_info(user_code)
        api_logger.debug(
            f"Authentication status checked: user authenticated as {user_code}"
        )
        return {
            "authenticated": True,
            "auth_enabled": True,
            "user": {"company": user_info.get("company", "Unknown"), "code": user_code},
        }
    api_logger.debug("Authentication status checked: user not authenticated")
    return {"authenticated": False, "auth_enabled": True}


@router.post("/chat")
async def chat(
    request: Request,
    query: str = Body(..., embed=True),
    user_code: str = Depends(require_auth),
):
    """
    Handle chat queries from authenticated users.

    Processes the query, logs interaction details, and returns the chat completion result.
    """
    user_info = get_user_info(user_code)
    company = user_info.get("company", "Unknown")
    session_token = request.cookies.get("session_token")

    api_logger.info(
        f"Received chat query from {company} (user_code: {user_code}), query length: {len(query)}"
    )

    # Prepare user metadata for LangSmith
    user_metadata = {"user_code": user_code, "company": company}

    # Measure response time
    start_time = time.time()

    # Call with langsmith_extra parameter
    result = get_chat_completion(
        query,
        user_metadata=user_metadata,
    )

    response_time = time.time() - start_time

    # Optional analytics must not prevent delivery of a successful chat response.
    try:
        response_text = ""
        sources = result.get("sources", [])
        if result.get("answer") and result["answer"].get("answer"):
            response_text = result["answer"]["answer"]
        elif result.get("answer"):
            response_text = str(result["answer"])

        advanced_analytics.log_chat_interaction_advanced(
            user_code=user_code,
            company=company,
            query=query,
            response=response_text,
            response_time=response_time,
            sources=sources,
            session_token=session_token or "",
            metadata={"llm_model": "gpt-4o-mini", "pipeline": "RAG"},
        )
    except Exception:
        api_logger.warning(
            "Analytics recording failed; returning chat response", exc_info=True
        )

    api_logger.info(f"Chat response generated for {company} in {response_time:.2f}s")
    return result


@router.post("/summary")
async def summary(
    request: Request,
    style: str = Query("bullet"),
    user_code: str = Depends(require_auth),
):
    """
    Generate a summary in the specified style for authenticated users.

    Logs summary generation details and returns the summary result.
    """
    user_info = get_user_info(user_code)
    company = user_info.get("company", "Unknown")

    api_logger.info(
        f"Summary request received from {company} (user_code: {user_code}), style: {style}"
    )

    # Prepare user metadata for LangSmith
    user_metadata = {"user_code": user_code, "company": company}

    result = get_auto_summary(style, user_metadata=user_metadata)

    summary_text = result.get("summary_md", "")

    # Advanced logging
    advanced_analytics.log_summary_request_advanced(
        user_code=user_code,
        company=company,
        style=style,
        summary_text=summary_text,
        metadata={"llm_model": "gpt-4o-mini", "pipeline": "Summarization"},
    )

    api_logger.info(
        f"Summary generated for {company}, length: {len(summary_text)} characters"
    )
    return result


@router.post("/job-match")
async def job_match_endpoint(request: Request, user_code: str = Depends(require_auth)):
    """
    Perform job description matching analysis for authenticated users.

    Accepts JSON with text input, processes the job description,
    performs analysis, and returns the matching results.
    """
    user_info = get_user_info(user_code)
    company = user_info.get("company", "Unknown")
    session_token = request.cookies.get("session_token")

    api_logger.info(
        f"Job matching request received from {company} (user_code: {user_code})"
    )

    try:
        # Parse JSON data for text input only
        body = await request.json()
        text_input = body.get("text")

        # Process job description from text only
        job_description = process_job_description(text=str(text_input))

        # Prepare user metadata for LangSmith
        user_metadata = {"user_code": user_code, "company": company}

        # Measure response time
        start_time = time.time()

        # Perform job matching analysis
        analysis_result = analyze_job_match(job_description, user_metadata)

        response_time = time.time() - start_time

        # Advanced logging
        advanced_analytics.log_job_matching_advanced(
            user_code=user_code,
            company=company,
            job_source=analysis_result.get("job_source", "text_input"),
            analysis_text=analysis_result["analysis"],
            response_time=response_time,
            session_token=session_token or "",
            metadata={"llm_model": "gpt-4o-mini", "pipeline": "JobMatching"},
        )

        api_logger.info(
            f"Job matching analysis completed for {company} in {response_time:.2f}s"
        )

        return {
            "analysis": analysis_result["analysis"],
            "source": analysis_result.get("job_source", "text_input"),
            "timestamp": analysis_result.get("match_timestamp"),
            "relevant_sections": analysis_result.get("relevant_sections", []),
        }

    except ValueError as e:
        api_logger.error(f"Job matching validation error for {company}: {e!s}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        api_logger.exception("Job matching processing error")
        raise HTTPException(status_code=500, detail="Job matching analysis failed")
