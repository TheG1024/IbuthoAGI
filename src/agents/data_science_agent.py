"""
Data Science agent implementation for data analysis and machine learning tasks.
"""
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from ..core.agent import Agent, AgentRole, AgentCapability
from ..utils.memory_manager import MemoryManager

class DataSciencePrompt:
    """Prompts for data science operations."""
    
    DATA_ANALYSIS = PromptTemplate(
        input_variables=["data_description", "analysis_requirements"],
        template="""
        Analyze the following dataset:
        
        Data Description: {data_description}
        Analysis Requirements: {analysis_requirements}
        
        Provide:
        1. Statistical analysis
        2. Data quality assessment
        3. Feature importance
        4. Correlation analysis
        5. Recommendations
        
        Format your response as a JSON object.
        """
    )
    
    MODEL_SELECTION = PromptTemplate(
        input_variables=["problem_type", "data_characteristics", "requirements"],
        template="""
        Recommend ML models for the following scenario:
        
        Problem Type: {problem_type}
        Data Characteristics: {data_characteristics}
        Requirements: {requirements}
        
        Provide:
        1. Recommended models
        2. Pros and cons
        3. Implementation approach
        4. Evaluation metrics
        5. Optimization strategies
        
        Format your response as a JSON object.
        """
    )

class DataScienceAgent(Agent):
    """Agent specialized in data analysis and machine learning."""
    
    def __init__(
        self,
        agent_id: str,
        memory_manager: MemoryManager,
        llm_chain: Optional[LLMChain] = None,
        embeddings_model: Optional[OpenAIEmbeddings] = None
    ):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.SPECIALIST,
            capabilities=[
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.MACHINE_LEARNING
            ],
            llm=llm_chain.llm if llm_chain else None
        )
        self.memory_manager = memory_manager
        self.llm_chain = llm_chain
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.model_store = {}
        self.analysis_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def analyze_data(
        self,
        data: Union[pd.DataFrame, str],
        analysis_requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive data analysis."""
        # Generate session ID
        session_id = f"data_analysis_{datetime.now().timestamp()}"
        
        # Load data if path provided
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        # Basic statistics
        stats = {
            "shape": data.shape,
            "dtypes": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "numerical_stats": data.describe().to_dict(),
            "categorical_stats": {
                col: data[col].value_counts().to_dict()
                for col in data.select_dtypes(include=['object']).columns
            }
        }
        
        # Correlation analysis for numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        correlations = data[numerical_cols].corr().to_dict()
        
        # Store analysis results
        analysis_results = {
            "basic_stats": stats,
            "correlations": correlations,
            "requirements_analysis": await self._analyze_requirements(
                data,
                analysis_requirements
            )
        }
        
        # Store session
        self.analysis_sessions[session_id] = {
            "timestamp": datetime.now(),
            "data_shape": data.shape,
            "analysis_requirements": analysis_requirements,
            "results": analysis_results
        }
        
        return {
            "session_id": session_id,
            "analysis": analysis_results
        }
    
    async def train_model(
        self,
        data: Union[pd.DataFrame, str],
        target_column: str,
        model_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """Train a machine learning model."""
        try:
            # Load data if path provided
            if isinstance(data, str):
                data = pd.read_csv(data)
            
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Handle categorical variables
            X = pd.get_dummies(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Get model class
            model_class = self._get_model_class(model_type)
            
            # Initialize and train model
            model = model_class(**(parameters or {}))
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Generate model ID and store model
            model_id = f"{model_type}_{datetime.now().timestamp()}"
            self.model_store[model_id] = {
                "model": model,
                "scaler": scaler,
                "feature_names": X.columns.tolist(),
                "target_column": target_column,
                "parameters": parameters,
                "metrics": {
                    "train_score": train_score,
                    "test_score": test_score
                }
            }
            
            return {
                "model_id": model_id,
                "metrics": {
                    "train_score": train_score,
                    "test_score": test_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            return {
                "error": str(e)
            }
    
    async def predict(
        self,
        model_id: str,
        data: Union[pd.DataFrame, str]
    ) -> Dict[str, Any]:
        """Make predictions using a trained model."""
        try:
            model_info = self.model_store.get(model_id)
            if not model_info:
                return {"error": "Model not found"}
            
            # Load data if path provided
            if isinstance(data, str):
                data = pd.read_csv(data)
            
            # Prepare features
            if isinstance(data, pd.DataFrame):
                # Ensure all required features are present
                missing_features = set(model_info["feature_names"]) - set(data.columns)
                if missing_features:
                    return {"error": f"Missing features: {missing_features}"}
                
                # Select and order features
                X = data[model_info["feature_names"]]
            else:
                return {"error": "Invalid data format"}
            
            # Scale features
            X_scaled = model_info["scaler"].transform(X)
            
            # Make predictions
            predictions = model_info["model"].predict(X_scaled)
            
            return {
                "predictions": predictions.tolist(),
                "model_id": model_id
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return {
                "error": str(e)
            }
    
    async def save_model(
        self,
        model_id: str,
        path: str
    ) -> Dict[str, Any]:
        """Save a trained model to disk."""
        try:
            model_info = self.model_store.get(model_id)
            if not model_info:
                return {"error": "Model not found"}
            
            # Save model and related information
            joblib.dump(model_info, path)
            
            return {
                "message": f"Model saved successfully to {path}",
                "model_id": model_id
            }
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            return {
                "error": str(e)
            }
    
    async def load_model(
        self,
        path: str
    ) -> Dict[str, Any]:
        """Load a saved model from disk."""
        try:
            # Load model and related information
            model_info = joblib.load(path)
            
            # Generate new model ID
            model_id = f"loaded_{datetime.now().timestamp()}"
            
            # Store model
            self.model_store[model_id] = model_info
            
            return {
                "message": "Model loaded successfully",
                "model_id": model_id
            }
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            return {
                "error": str(e)
            }
    
    async def _analyze_requirements(
        self,
        data: pd.DataFrame,
        requirements: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze data based on specific requirements."""
        if not self.llm_chain or not requirements:
            return {}
            
        response = await self.llm_chain.arun(
            DataSciencePrompt.DATA_ANALYSIS,
            data_description=data.describe().to_json(),
            analysis_requirements=json.dumps(requirements)
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse data analysis response")
            return {}
    
    def _get_model_class(self, model_type: str):
        """Get the appropriate model class based on type."""
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC, SVR
        
        model_map = {
            "linear_regression": LinearRegression,
            "logistic_regression": LogisticRegression,
            "random_forest_classifier": RandomForestClassifier,
            "random_forest_regressor": RandomForestRegressor,
            "svc": SVC,
            "svr": SVR
        }
        
        model_class = model_map.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model_class
    
    def get_session_details(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Get detailed information about an analysis session."""
        session = self.analysis_sessions.get(session_id, {})
        return {
            "timestamp": session.get("timestamp"),
            "data_shape": session.get("data_shape"),
            "requirements": session.get("analysis_requirements"),
            "results": session.get("results")
        }
    
    def get_model_info(
        self,
        model_id: str
    ) -> Dict[str, Any]:
        """Get information about a trained model."""
        model_info = self.model_store.get(model_id, {})
        if not model_info:
            return {"error": "Model not found"}
        
        return {
            "feature_names": model_info.get("feature_names"),
            "target_column": model_info.get("target_column"),
            "parameters": model_info.get("parameters"),
            "metrics": model_info.get("metrics")
        }
