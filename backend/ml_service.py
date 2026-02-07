"""
Machine Learning Service for Bug Prediction
Loads trained model and makes predictions
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler

from utils import extract_features_from_code

logger = logging.getLogger(__name__)

class MLService:
    """Machine Learning service for bug prediction"""
    
    def __init__(self, model_path: str = "ml/model.pkl"):
        """
        Initialize ML service with trained model
        
        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.request_count = 0
        self.success_count = 0
        
        try:
            self.load_model(model_path)
            logger.info("ML service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ML service: {str(e)}")
            self.model = self.create_fallback_model()
    
    def load_model(self, model_path: str):
        """Load trained model from file"""
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                logger.warning(f"Model file not found at {model_path}, using fallback")
                self.model = self.create_fallback_model()
                return
            
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', [])
            
            if self.model is None:
                raise ValueError("Model not found in loaded data")
                
            logger.info(f"Model loaded successfully. Features: {self.feature_names}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def create_fallback_model(self):
        """Create a simple fallback model when trained model is unavailable"""
        logger.info("Creating fallback ML model")
        
        class FallbackModel:
            def predict_proba(self, X):
                # Simple heuristic based on code complexity
                n_samples = X.shape[0] if len(X.shape) > 1 else 1
                predictions = []
                
                for i in range(n_samples):
                    if len(X.shape) > 1:
                        features = X[i]
                    else:
                        features = X
                    
                    # Simple heuristic: more features = higher bug probability
                    complexity_score = np.sum(features[:5]) / 100  # First 5 features
                    prob = min(0.9, max(0.1, complexity_score))
                    predictions.append([1-prob, prob])
                
                return np.array(predictions)
            
            def predict(self, X):
                probs = self.predict_proba(X)
                return (probs[:, 1] > 0.5).astype(int)
        
        return FallbackModel()
    
    def extract_features(self, code: str) -> np.ndarray:
        """
        Extract features from code for ML prediction
        
        Args:
            code: Source code string
            
        Returns:
            numpy array of features
        """
        self.request_count += 1
        
        try:
            # Extract basic features
            features_dict = extract_features_from_code(code)
            
            # Convert to array in correct order
            if self.feature_names:
                features = [features_dict.get(name, 0) for name in self.feature_names]
            else:
                # Use default feature order
                features = [
                    features_dict.get('loc', 0),
                    features_dict.get('cyclomatic_complexity', 0),
                    features_dict.get('halstead_volume', 0),
                    features_dict.get('num_functions', 0),
                    features_dict.get('num_loops', 0),
                    features_dict.get('num_conditionals', 0),
                    features_dict.get('num_try_except', 0),
                    features_dict.get('num_null_checks', 0),
                    features_dict.get('avg_line_length', 0),
                    features_dict.get('comment_density', 0)
                ]
            
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features if scaler is available
            if self.scaler:
                features_array = self.scaler.transform(features_array)
            
            self.success_count += 1
            return features_array
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            # Return zero features as fallback
            return np.zeros((1, 10))
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predict bug probability
        
        Args:
            features: Feature array
            
        Returns:
            Bug probability (0 to 1)
        """
        try:
            if self.model is None:
                # Fallback prediction based on feature magnitude
                return min(0.9, np.mean(features) / 10)
            
            # Get probability from model
            probabilities = self.model.predict_proba(features)
            bug_probability = probabilities[0, 1]  # Probability of class 1 (bug)
            
            # Ensure probability is within bounds
            return max(0.0, min(1.0, float(bug_probability)))
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            # Fallback: return moderate probability
            return 0.5
    
    def get_severity(self, probability: float) -> str:
        """
        Determine bug severity based on probability
        
        Args:
            probability: Bug probability (0 to 1)
            
        Returns:
            Severity level: 'critical', 'high', 'medium', or 'low'
        """
        if probability > 0.8:
            return "critical"
        elif probability > 0.6:
            return "high"
        elif probability > 0.4:
            return "medium"
        else:
            return "low"
    
    def generate_explanation(self, features: np.ndarray, probability: float, severity: str) -> str:
        """
        Generate human-readable explanation based on features
        
        Args:
            features: Feature array
            probability: Bug probability
            severity: Severity level
            
        Returns:
            Explanation string
        """
        try:
            if features.size == 0:
                return "Unable to analyze code features."
            
            # Get feature values
            feat_values = features.flatten()
            
            explanations = []
            
            # Analyze based on feature values
            if len(feat_values) > 0 and feat_values[0] > 100:  # LOC
                explanations.append(f"• Code is quite long ({int(feat_values[0])} lines), which increases complexity")
            
            if len(feat_values) > 1 and feat_values[1] > 10:  # Cyclomatic complexity
                explanations.append(f"• High cyclomatic complexity ({feat_values[1]:.1f}), indicating many decision paths")
            
            if len(feat_values) > 2 and feat_values[2] > 1000:  # Halstead volume
                explanations.append("• High Halstead volume, suggesting complex algorithms")
            
            if len(feat_values) > 3 and feat_values[3] > 5:  # Number of functions
                explanations.append(f"• Multiple functions ({int(feat_values[3])}), good for modularity but needs proper testing")
            
            if len(feat_values) > 4 and feat_values[4] > 3:  # Number of loops
                explanations.append(f"• Contains {int(feat_values[4])} loops - check for infinite loop conditions")
            
            if len(feat_values) > 5 and feat_values[5] > 5:  # Number of conditionals
                explanations.append(f"• Many conditional statements ({int(feat_values[5])}), which can lead to logic errors")
            
            # If no specific features stand out, give generic explanation
            if not explanations:
                if probability > 0.7:
                    explanations.append("• Code shows patterns commonly associated with bugs")
                elif probability > 0.4:
                    explanations.append("• Some concerning patterns detected")
                else:
                    explanations.append("• Code appears relatively clean")
            
            # Add probability and severity
            explanation = f"Bug Probability: {probability:.1%}\nSeverity: {severity.upper()}\n\n"
            explanation += "Analysis:\n" + "\n".join(explanations)
            
            # Add recommendations
            explanation += "\n\nRecommendations:\n"
            if severity == "critical":
                explanation += "• Review code thoroughly before deployment\n"
                explanation += "• Add comprehensive error handling\n"
                explanation += "• Consider code refactoring\n"
            elif severity == "high":
                explanation += "• Test edge cases thoroughly\n"
                explanation += "• Add input validation\n"
                explanation += "• Review loop conditions\n"
            elif severity == "medium":
                explanation += "• Add more comments for complex logic\n"
                explanation += "• Consider breaking down large functions\n"
                explanation += "• Add unit tests\n"
            else:
                explanation += "• Continue with current practices\n"
                explanation += "• Regular code reviews\n"
                explanation += "• Consider adding more test cases\n"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return f"Bug Probability: {probability:.1%}\nSeverity: {severity.upper()}\n\nAnalysis unavailable."
    
    def calculate_metrics(self, code: str) -> Dict[str, Any]:
        """
        Calculate code metrics
        
        Args:
            code: Source code string
            
        Returns:
            Dictionary of metrics
        """
        try:
            features = extract_features_from_code(code)
            
            return {
                "loc": features.get("loc", 0),
                "complexity": features.get("cyclomatic_complexity", 0),
                "num_functions": features.get("num_functions", 0),
                "num_loops": features.get("num_loops", 0),
                "num_conditionals": features.get("num_conditionals", 0),
                "issues": self._estimate_issues(features),
                "maintainability": self._calculate_maintainability(features)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                "loc": 0,
                "complexity": 0,
                "num_functions": 0,
                "num_loops": 0,
                "num_conditionals": 0,
                "issues": 0,
                "maintainability": 0
            }
    
    def _estimate_issues(self, features: Dict[str, float]) -> int:
        """Estimate number of potential issues"""
        issues = 0
        
        if features.get("cyclomatic_complexity", 0) > 10:
            issues += 1
        if features.get("num_loops", 0) > 3:
            issues += 1
        if features.get("num_conditionals", 0) > 5:
            issues += 1
        if features.get("loc", 0) > 50 and features.get("num_functions", 0) == 0:
            issues += 1  # Large function without decomposition
        
        return issues
    
    def _calculate_maintainability(self, features: Dict[str, float]) -> float:
        """Calculate maintainability index (simplified)"""
        loc = features.get("loc", 1)
        complexity = features.get("cyclomatic_complexity", 1)
        comment_density = features.get("comment_density", 0)
        
        # Simplified maintainability calculation
        maintainability = 100 - (complexity * 2) - (loc / 10) + (comment_density * 10)
        return max(0, min(100, maintainability))
    
    def get_accuracy(self) -> float:
        """Get model accuracy (simulated for demo)"""
        if self.model is None:
            return 0.85  # Fallback accuracy
        
        # In real implementation, this would return actual accuracy
        return 0.87  # Simulated accuracy
    
    def is_ready(self) -> bool:
        """Check if ML service is ready"""
        return self.model is not None

# Singleton instance
ml_service = MLService()