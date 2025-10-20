"""
Pod Forecasting Model
Machine learning model to predict Kubernetes pod requirements
"""
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from loguru import logger


class PodForecastingModel:
    """Machine learning model for forecasting pod requirements"""
    
    def __init__(self):
        """Initialize the forecasting model"""
        self.frontend_model = None
        self.backend_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['GMV', 'Users', 'Marketing_Cost', 'DayOfWeek', 'DayOfMonth', 'Month']
        self.feature_names_used = None  # Will be set during training
        self.metrics = {}
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from the data
        
        Args:
            df: Input DataFrame with Date, GMV, Users, Marketing_Cost
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Ensure Date is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Time-based features
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['DayOfMonth'] = df['Date'].dt.day
            df['Month'] = df['Date'].dt.month
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Interaction features
        if all(col in df.columns for col in ['GMV', 'Users']):
            df['GMV_per_User'] = df['GMV'] / (df['Users'] + 1)  # +1 to avoid division by zero
        
        if all(col in df.columns for col in ['Marketing_Cost', 'Users']):
            df['Marketing_per_User'] = df['Marketing_Cost'] / (df['Users'] + 1)
        
        if all(col in df.columns for col in ['GMV', 'Marketing_Cost']):
            df['ROAS'] = df['GMV'] / (df['Marketing_Cost'] + 1)  # Return on Ad Spend
        
        # Rolling averages (if enough data)
        if len(df) >= 7:
            df['GMV_7day_avg'] = df['GMV'].rolling(window=7, min_periods=1).mean()
            df['Users_7day_avg'] = df['Users'].rolling(window=7, min_periods=1).mean()
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, training: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix for training or prediction
        
        Args:
            df: DataFrame with engineered features
            training: If True, determine which features to use. If False, use stored features.
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        if training:
            # During training: select all available features
            available_features = [col for col in self.feature_names if col in df.columns]
            
            # Add interaction features if available (but skip rolling averages for single predictions)
            interaction_features = ['GMV_per_User', 'Marketing_per_User', 'ROAS', 'IsWeekend']
            for feat in interaction_features:
                if feat in df.columns:
                    available_features.append(feat)
            
            # Store for future predictions
            self.feature_names_used = available_features
        else:
            # During prediction: use exact same features as training
            available_features = self.feature_names_used
        
        # Ensure all features exist, fill missing with 0
        X = df[available_features].fillna(0).values
        
        return X, available_features
    
    def train(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the forecasting models
        
        Args:
            historical_data: DataFrame with historical data including pod counts
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting model training...")
        
        # Engineer features
        df = self._engineer_features(historical_data)
        
        # Prepare features and targets
        X, feature_names = self.prepare_features(df, training=True)
        y_frontend = df['Frontend_Pods'].values
        y_backend = df['Backend_Pods'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        X_train, X_val, y_fe_train, y_fe_val, y_be_train, y_be_val = train_test_split(
            X_scaled, y_frontend, y_backend, test_size=0.2, random_state=42
        )
        
        # Train frontend model
        logger.info("Training frontend pod model...")
        self.frontend_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.frontend_model.fit(X_train, y_fe_train)
        
        # Train backend model
        logger.info("Training backend pod model...")
        self.backend_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.backend_model.fit(X_train, y_be_train)
        
        # Evaluate models
        fe_pred = self.frontend_model.predict(X_val)
        be_pred = self.backend_model.predict(X_val)
        
        self.metrics = {
            'frontend_mae': mean_absolute_error(y_fe_val, fe_pred),
            'frontend_rmse': np.sqrt(mean_squared_error(y_fe_val, fe_pred)),
            'frontend_r2': r2_score(y_fe_val, fe_pred),
            'backend_mae': mean_absolute_error(y_be_val, be_pred),
            'backend_rmse': np.sqrt(mean_squared_error(y_be_val, be_pred)),
            'backend_r2': r2_score(y_be_val, be_pred),
        }
        
        self.is_trained = True
        self.feature_names_used = feature_names
        
        logger.info(f"Model training complete. Frontend R²: {self.metrics['frontend_r2']:.3f}, Backend R²: {self.metrics['backend_r2']:.3f}")
        
        return self.metrics
    
    def predict(self, budget_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict pod requirements for budget data
        
        Args:
            budget_data: DataFrame with budget data (GMV, Users, Marketing_Cost, Date)
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Engineer features
        df = self._engineer_features(budget_data)
        
        # Prepare features (use same features as training)
        X, _ = self.prepare_features(df, training=False)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        frontend_preds = self.frontend_model.predict(X_scaled)
        backend_preds = self.backend_model.predict(X_scaled)
        
        # Round to nearest integer (can't have fractional pods)
        frontend_preds = np.round(frontend_preds).astype(int)
        backend_preds = np.round(backend_preds).astype(int)
        
        # Ensure minimum of 1 pod
        frontend_preds = np.maximum(frontend_preds, 1)
        backend_preds = np.maximum(backend_preds, 1)
        
        # Create results DataFrame
        results = budget_data[['Date', 'GMV', 'Users', 'Marketing_Cost']].copy()
        results['Frontend_Pods'] = frontend_preds
        results['Backend_Pods'] = backend_preds
        results['Total_Pods'] = frontend_preds + backend_preds
        
        # Calculate confidence scores (inverse of validation error)
        fe_confidence = 1 / (1 + self.metrics['frontend_mae'])
        be_confidence = 1 / (1 + self.metrics['backend_mae'])
        results['Confidence_Score'] = (fe_confidence + be_confidence) / 2
        
        return results
    
    def save_model(self, path: str):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'frontend_model': self.frontend_model,
            'backend_model': self.backend_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names_used,
            'metrics': self.metrics
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model from disk"""
        model_data = joblib.load(path)
        self.frontend_model = model_data['frontend_model']
        self.backend_model = model_data['backend_model']
        self.scaler = model_data['scaler']
        self.feature_names_used = model_data['feature_names']
        self.metrics = model_data['metrics']
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
