"""
Advanced analytics module with AI-powered anomaly detection and predictive analytics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

@dataclass
class AnomalyDetectionResult:
    """Results from anomaly detection."""
    is_anomaly: bool
    anomaly_score: float
    affected_metrics: List[str]
    timestamp: datetime
    severity: str
    recommended_actions: List[str]

@dataclass
class PredictionResult:
    """Results from predictive analytics."""
    predicted_values: np.ndarray
    confidence_intervals: np.ndarray
    timestamps: List[datetime]
    metric_name: str
    model_confidence: float

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data."""
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 10
    ):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return x, y

class LSTMAnomalyDetector(nn.Module):
    """LSTM-based anomaly detector."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class AnomalyDetector:
    """AI-powered anomaly detection system."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.lstm_model = None
        self.threshold = 3.0  # Standard deviations for LSTM-based detection
    
    def train_models(
        self,
        data: pd.DataFrame,
        sequence_length: int = 10
    ) -> None:
        """Train both isolation forest and LSTM models."""
        # Train Isolation Forest
        scaled_data = self.scaler.fit_transform(data)
        self.isolation_forest.fit(scaled_data)
        
        # Train LSTM
        dataset = TimeSeriesDataset(scaled_data, sequence_length)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True
        )
        
        self.lstm_model = LSTMAnomalyDetector(
            input_size=data.shape[1]
        )
        optimizer = torch.optim.Adam(self.lstm_model.parameters())
        criterion = nn.MSELoss()
        
        self.lstm_model.train()
        for epoch in range(10):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.lstm_model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
    
    def detect_anomalies(
        self,
        current_data: pd.DataFrame
    ) -> List[AnomalyDetectionResult]:
        """Detect anomalies using both models."""
        results = []
        scaled_data = self.scaler.transform(current_data)
        
        # Isolation Forest Detection
        if_predictions = self.isolation_forest.predict(scaled_data)
        
        # LSTM Detection
        self.lstm_model.eval()
        with torch.no_grad():
            sequence = torch.FloatTensor(scaled_data[-10:]).unsqueeze(0)
            lstm_pred = self.lstm_model(sequence)
            
            # Calculate reconstruction error
            mse = nn.MSELoss(reduction='none')
            reconstruction_error = mse(
                lstm_pred,
                torch.FloatTensor(scaled_data[-1:])
            ).numpy()
        
        # Combine results
        for i in range(len(current_data)):
            if if_predictions[i] == -1 or reconstruction_error[i] > self.threshold:
                affected_metrics = [
                    col for col, error in zip(
                        current_data.columns,
                        reconstruction_error[i]
                    )
                    if error > self.threshold
                ]
                
                severity = 'High' if reconstruction_error[i] > self.threshold * 2 else 'Medium'
                
                results.append(AnomalyDetectionResult(
                    is_anomaly=True,
                    anomaly_score=float(reconstruction_error[i]),
                    affected_metrics=affected_metrics,
                    timestamp=datetime.now(),
                    severity=severity,
                    recommended_actions=self._generate_recommendations(
                        affected_metrics,
                        severity
                    )
                ))
        
        return results
    
    def _generate_recommendations(
        self,
        affected_metrics: List[str],
        severity: str
    ) -> List[str]:
        """Generate recommended actions based on anomalies."""
        recommendations = []
        
        for metric in affected_metrics:
            if 'cpu' in metric.lower():
                recommendations.extend([
                    "Check for resource-intensive processes",
                    "Consider scaling up compute resources"
                ])
            elif 'memory' in metric.lower():
                recommendations.extend([
                    "Investigate memory leaks",
                    "Consider increasing memory allocation"
                ])
            elif 'api' in metric.lower():
                recommendations.extend([
                    "Check API endpoint health",
                    "Verify API rate limits"
                ])
            elif 'error' in metric.lower():
                recommendations.extend([
                    "Review error logs",
                    "Check system health metrics"
                ])
        
        if severity == 'High':
            recommendations.append("Immediate attention required")
        
        return list(set(recommendations))

class PredictiveAnalytics:
    """AI-powered predictive analytics system."""
    def __init__(self):
        self.prophet_models: Dict[str, Prophet] = {}
        self.rf_models: Dict[str, RandomForestRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
    
    def train_models(
        self,
        data: Dict[str, pd.DataFrame],
        forecast_horizon: int = 24
    ) -> None:
        """Train prediction models for each metric."""
        for metric_name, metric_data in data.items():
            # Prophet Model
            prophet_df = pd.DataFrame({
                'ds': metric_data.index,
                'y': metric_data.values
            })
            self.prophet_models[metric_name] = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True
            )
            self.prophet_models[metric_name].fit(prophet_df)
            
            # Random Forest Model
            self.scalers[metric_name] = StandardScaler()
            scaled_data = self.scalers[metric_name].fit_transform(
                metric_data.values.reshape(-1, 1)
            )
            
            X, y = self._create_sequences(scaled_data, 10)
            self.rf_models[metric_name] = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            self.rf_models[metric_name].fit(X, y)
    
    def predict(
        self,
        metric_name: str,
        horizon_hours: int = 24
    ) -> PredictionResult:
        """Generate predictions using both models."""
        # Prophet Prediction
        future_dates = self.prophet_models[metric_name].make_future_dataframe(
            periods=horizon_hours,
            freq='H'
        )
        prophet_forecast = self.prophet_models[metric_name].predict(future_dates)
        
        # Random Forest Prediction
        last_sequence = self.scalers[metric_name].transform(
            prophet_forecast['yhat'].values[-horizon_hours-10:-horizon_hours].reshape(-1, 1)
        )
        rf_prediction = self.rf_models[metric_name].predict(
            last_sequence.reshape(1, -1)
        )
        
        # Combine predictions
        combined_prediction = (
            prophet_forecast['yhat'].values[-horizon_hours:] +
            self.scalers[metric_name].inverse_transform(
                rf_prediction.reshape(-1, 1)
            ).flatten()
        ) / 2
        
        confidence_intervals = np.column_stack([
            prophet_forecast['yhat_lower'].values[-horizon_hours:],
            prophet_forecast['yhat_upper'].values[-horizon_hours:]
        ])
        
        return PredictionResult(
            predicted_values=combined_prediction,
            confidence_intervals=confidence_intervals,
            timestamps=future_dates['ds'].values[-horizon_hours:].tolist(),
            metric_name=metric_name,
            model_confidence=self._calculate_confidence(
                prophet_forecast['yhat'].values[-horizon_hours:],
                rf_prediction
            )
        )
    
    def _create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def _calculate_confidence(
        self,
        prophet_pred: np.ndarray,
        rf_pred: np.ndarray
    ) -> float:
        """Calculate prediction confidence based on model agreement."""
        difference = np.abs(prophet_pred - rf_pred)
        max_diff = np.max(prophet_pred) - np.min(prophet_pred)
        return 1 - (np.mean(difference) / max_diff)
