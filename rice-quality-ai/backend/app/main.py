"""
FastAPI Main Application for Rice Quality AI.
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path so ml and backend can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("rice_quality_ai")

app = FastAPI(
    title="Rice Quality AI API",
    description="Automated Image-Based Raw Milled Rice Grain Quality Analysis System",
    version="1.0.0",
)

# Enable CORS for React frontend (Vite default port 5173, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")
# Mount directly at root for convenience (e.g. /health, /models, /standards)
app.include_router(router)

# Mount frontend dist static files if built
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {
            "message": "Rice Quality AI API is running.",
            "docs": "/docs",
            "health": "/health",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
