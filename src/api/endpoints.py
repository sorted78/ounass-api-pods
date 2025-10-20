"""
API Endpoints
FastAPI endpoints for pod forecasting
"""
from datetime import datetime, date, timedelta
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger

from ..services.sheets_service import GoogleSheetsService
from ..models.forecasting import PodForecastingModel


# Pydantic models for API responses
class PodPrediction(BaseModel):
    """Single day pod prediction"""
    date: date
    frontend_pods: int = Field(..., description="Number of frontend pods required")
    backend_pods: int = Field(..., description="Number of backend pods required")
    total_pods: int = Field(..., description="Total pods required")
    confidence_score: float = Field(..., description="Prediction confidence score")
    metrics: dict = Field(..., description="Business metrics used for prediction")


class PredictionRange(BaseModel):
    """Range of pod predictions"""
    start_date: date
    end_date: date
    predictions: List[PodPrediction]
    summary: dict


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = {"protected_namespaces": ()}
    
    status: str
    model_trained: bool
    timestamp: datetime


class TrainingMetrics(BaseModel):
    """Model training metrics"""
    frontend_mae: float
    frontend_rmse: float
    frontend_r2: float
    backend_mae: float
    backend_rmse: float
    backend_r2: float


# Router
router = APIRouter(prefix="/api/v1", tags=["forecasting"])

# Global instances (will be initialized in main.py)
sheets_service: Optional[GoogleSheetsService] = None
forecasting_model: Optional[PodForecastingModel] = None


def initialize_services(sheets_svc: GoogleSheetsService, model: PodForecastingModel):
    """Initialize global service instances"""
    global sheets_service, forecasting_model
    sheets_service = sheets_svc
    forecasting_model = model


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy",
        model_trained=forecasting_model.is_trained if forecasting_model else False,
        timestamp=datetime.now()
    )


@router.post("/train", response_model=TrainingMetrics)
async def train_model():
    """
    Trigger model training with latest data from Google Sheets
    """
    try:
        logger.info("Fetching historical data for training...")
        historical_data = sheets_service.get_historical_data()
        
        if len(historical_data) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient historical data. Need at least 10 records, got {len(historical_data)}"
            )
        
        logger.info(f"Training model with {len(historical_data)} historical records...")
        metrics = forecasting_model.train(historical_data)
        
        return TrainingMetrics(**metrics)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/forecast/daily", response_model=PodPrediction)
async def get_daily_forecast(
    target_date: Optional[date] = Query(None, description="Target date for prediction (defaults to tomorrow)")
):
    """
    Get pod prediction for a specific day
    """
    if not forecasting_model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Please call /api/v1/train first."
        )
    
    try:
        # Default to tomorrow if no date provided
        if target_date is None:
            target_date = date.today() + timedelta(days=1)
        
        # Fetch all data
        data = sheets_service.get_all_data()
        budget_data = data['budget']
        
        # Filter for the target date
        budget_data['Date'] = budget_data['Date'].dt.date
        target_data = budget_data[budget_data['Date'] == target_date]
        
        if len(target_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No budget data found for date {target_date}"
            )
        
        # Make prediction
        prediction = forecasting_model.predict(target_data)
        
        result = prediction.iloc[0]
        
        return PodPrediction(
            date=result['Date'].date() if hasattr(result['Date'], 'date') else result['Date'],
            frontend_pods=int(result['Frontend_Pods']),
            backend_pods=int(result['Backend_Pods']),
            total_pods=int(result['Total_Pods']),
            confidence_score=float(result['Confidence_Score']),
            metrics={
                "gmv": float(result['GMV']),
                "users": int(result['Users']),
                "marketing_cost": float(result['Marketing_Cost'])
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/forecast/range", response_model=PredictionRange)
async def get_forecast_range(
    start_date: Optional[date] = Query(None, description="Start date for predictions"),
    end_date: Optional[date] = Query(None, description="End date for predictions"),
    days: Optional[int] = Query(None, description="Number of days to forecast (alternative to end_date)")
):
    """
    Get pod predictions for a date range
    """
    if not forecasting_model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Please call /api/v1/train first."
        )
    
    try:
        # Set defaults
        if start_date is None:
            start_date = date.today() + timedelta(days=1)
        
        if end_date is None and days is None:
            days = 30  # Default to 30 days
        
        if end_date is None:
            end_date = start_date + timedelta(days=days - 1)
        
        # Fetch budget data
        data = sheets_service.get_all_data()
        budget_data = data['budget']
        
        # Filter for date range
        budget_data['Date'] = budget_data['Date'].dt.date
        range_data = budget_data[
            (budget_data['Date'] >= start_date) & 
            (budget_data['Date'] <= end_date)
        ].sort_values('Date')
        
        if len(range_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No budget data found for date range {start_date} to {end_date}"
            )
        
        # Make predictions
        predictions_df = forecasting_model.predict(range_data)
        
        # Convert to list of predictions
        predictions = []
        for _, row in predictions_df.iterrows():
            predictions.append(PodPrediction(
                date=row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'],
                frontend_pods=int(row['Frontend_Pods']),
                backend_pods=int(row['Backend_Pods']),
                total_pods=int(row['Total_Pods']),
                confidence_score=float(row['Confidence_Score']),
                metrics={
                    "gmv": float(row['GMV']),
                    "users": int(row['Users']),
                    "marketing_cost": float(row['Marketing_Cost'])
                }
            ))
        
        # Calculate summary statistics
        summary = {
            "avg_frontend_pods": float(predictions_df['Frontend_Pods'].mean()),
            "avg_backend_pods": float(predictions_df['Backend_Pods'].mean()),
            "avg_total_pods": float(predictions_df['Total_Pods'].mean()),
            "max_total_pods": int(predictions_df['Total_Pods'].max()),
            "min_total_pods": int(predictions_df['Total_Pods'].min()),
            "total_days": len(predictions)
        }
        
        return PredictionRange(
            start_date=start_date,
            end_date=end_date,
            predictions=predictions,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Range prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Range prediction failed: {str(e)}")


@router.get("/metrics", response_model=TrainingMetrics)
async def get_model_metrics():
    """
    Get current model performance metrics
    """
    if not forecasting_model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. No metrics available."
        )
    
    return TrainingMetrics(**forecasting_model.metrics)
