"""
Google Sheets Service
Handles fetching data from Google Sheets API
"""
import os
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from loguru import logger


class GoogleSheetsService:
    """Service for interacting with Google Sheets API"""
    
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets.readonly',
        'https://www.googleapis.com/auth/drive.readonly'
    ]
    
    def __init__(self, credentials_path: str, sheet_id: str):
        """
        Initialize Google Sheets service
        
        Args:
            credentials_path: Path to Google service account credentials JSON
            sheet_id: Google Sheet ID
        """
        self.credentials_path = credentials_path
        self.sheet_id = sheet_id
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Google Sheets"""
        try:
            credentials = Credentials.from_service_account_file(
                self.credentials_path,
                scopes=self.SCOPES
            )
            self.client = gspread.authorize(credentials)
            logger.info(f"Successfully connected to Google Sheets API")
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets: {e}")
            raise
    
    def fetch_data(self, sheet_name: str = "Sheet1") -> pd.DataFrame:
        """
        Fetch data from Google Sheet
        
        Args:
            sheet_name: Name of the sheet tab to fetch
            
        Returns:
            DataFrame with the sheet data
        """
        try:
            spreadsheet = self.client.open_by_key(self.sheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)
            
            # Get all values
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            
            # Convert date column to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Convert numeric columns
            numeric_columns = ['GMV', 'Users', 'Marketing_Cost', 'Frontend_Pods', 'Backend_Pods']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Fetched {len(df)} rows from sheet '{sheet_name}'")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data from Google Sheets: {e}")
            raise
    
    def get_historical_data(self, sheet_name: str = "Sheet1") -> pd.DataFrame:
        """
        Get historical data (rows with pod information)
        
        Args:
            sheet_name: Name of the sheet tab
            
        Returns:
            DataFrame with historical data only
        """
        df = self.fetch_data(sheet_name)
        
        # Filter rows where pod data exists
        historical = df[
            df['Frontend_Pods'].notna() & 
            df['Backend_Pods'].notna()
        ].copy()
        
        logger.info(f"Found {len(historical)} historical records with pod data")
        return historical
    
    def get_budget_data(self, sheet_name: str = "Sheet1") -> pd.DataFrame:
        """
        Get budgeted data (rows without pod information but with GMV, users, marketing)
        
        Args:
            sheet_name: Name of the sheet tab
            
        Returns:
            DataFrame with budget data only
        """
        df = self.fetch_data(sheet_name)
        
        # Filter rows where pod data doesn't exist but business metrics do
        budget = df[
            (df['Frontend_Pods'].isna() | df['Backend_Pods'].isna()) &
            df['GMV'].notna() &
            df['Users'].notna() &
            df['Marketing_Cost'].notna()
        ].copy()
        
        logger.info(f"Found {len(budget)} budget records without pod data")
        return budget
    
    def get_all_data(self, sheet_name: str = "Sheet1") -> Dict[str, pd.DataFrame]:
        """
        Get both historical and budget data
        
        Args:
            sheet_name: Name of the sheet tab
            
        Returns:
            Dictionary with 'historical' and 'budget' DataFrames
        """
        return {
            'historical': self.get_historical_data(sheet_name),
            'budget': self.get_budget_data(sheet_name)
        }
