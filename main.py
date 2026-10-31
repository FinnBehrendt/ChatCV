"""
ChatCV - Interactive Resume Assistant

A FastAPI application that transforms static resumes into intelligent,
conversational experiences using RAG (Retrieval-Augmented Generation).
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router as chat_router
from app.config import config
from app.utils.logging_config import main_logger

# Initialize logging
logger = main_logger

app = FastAPI(
    title="ChatCV-OSS",
    description="Interactive Resume Assistant powered by AI",
    version="1.0.0",
    docs_url="/docs" if config.environment == "development" else None,
    redoc_url="/redoc" if config.environment == "development" else None,
)

logger.info("ChatCV application starting up")

app.include_router(chat_router)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """
    Serve the main chat interface.

    Args:
        request: FastAPI request object

    Returns:
        HTMLResponse: Rendered chat.html template with candidate information
    """
    logger.debug("Serving main chat interface")
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "candidate_name": config.candidate.name,
            "candidate_email": config.candidate.email,
            "candidate_linkedin": config.candidate.linkedin,
            "candidate_github": config.candidate.github,
            "cv_path": "/resume",
        },
    )


@app.get("/resume", response_class=FileResponse)
def download_resume() -> FileResponse:
    """Publicly serve only the configured resume, never a user-selected path."""
    resume = Path(config.data.cv_path)
    if not resume.is_file():
        raise HTTPException(status_code=404, detail="Resume not found")
    return FileResponse(resume, media_type="application/pdf", filename="CV.pdf")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        Dict[str, str]: Status and service information
    """
    return {"status": "healthy", "service": "ChatCV-OSS"}


@app.on_event("startup")
async def startup_event() -> None:
    """Handle application startup tasks."""
    logger.info("ChatCV application startup completed")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Handle application shutdown tasks."""
    logger.info("ChatCV application shutting down")
