"""
Main FastAPI Application
OUNASS Kubernetes Pod Forecasting API
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from loguru import logger
import sys

from .api.endpoints import router, initialize_services
from .services.sheets_service import GoogleSheetsService
from .models.forecasting import PodForecastingModel


# Configuration
class Settings(BaseSettings):
    """Application settings"""
    google_sheet_id: str
    google_credentials_path: str = "./credentials.json"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"


# Initialize settings
settings = Settings()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.log_level
)
logger.add(
    "logs/api.log",
    rotation="1 day",
    retention="30 days",
    level=settings.log_level
)


# Global service instances
sheets_service = None
forecasting_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown
    """
    # Startup
    global sheets_service, forecasting_model
    
    logger.info("Starting OUNASS Pod Forecasting API...")
    
    try:
        # Initialize Google Sheets service
        logger.info("Initializing Google Sheets service...")
        sheets_service = GoogleSheetsService(
            credentials_path=settings.google_credentials_path,
            sheet_id=settings.google_sheet_id
        )
        
        # Initialize forecasting model
        logger.info("Initializing forecasting model...")
        forecasting_model = PodForecastingModel()
        
        # Initialize API endpoints with services
        initialize_services(sheets_service, forecasting_model)
        
        logger.info("API startup complete!")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        logger.warning("API will start but some features may not work until services are properly configured")
    
    yield
    
    # Shutdown
    logger.info("Shutting down OUNASS Pod Forecasting API...")


# Create FastAPI app
app = FastAPI(
    title="OUNASS Kubernetes Pod Forecasting API",
    description="Predict Kubernetes pod requirements based on business metrics",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OUNASS Kubernetes Pod Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
