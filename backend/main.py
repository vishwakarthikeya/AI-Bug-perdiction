"""
FastAPI Backend for AI Bug Predictor
Main application file with API endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import uvicorn

from ml_service import MLService
from ai_service import AIService
from schemas import (
    CodeAnalysisRequest,
    CodeAnalysisResponse,
    StudentChatRequest,
    StudentChatResponse,
    DeveloperFixRequest,
    DeveloperFixResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Bug Predictor API",
    description="API for bug prediction and code analysis using ML and AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ml_service = MLService()
ai_service = AIService()

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "AI Bug Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "API documentation",
            "/predict-bug": "POST - Analyze code for bugs",
            "/student-chat": "POST - AI explanation for students",
            "/developer-fix": "POST - Get AI-powered code fix"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "ml_service": ml_service.is_ready(),
            "ai_service": ai_service.is_ready()
        }
    }

@app.post("/predict-bug", response_model=CodeAnalysisResponse)
async def predict_bug(request: CodeAnalysisRequest):
    """
    Analyze code for potential bugs using ML model
    """
    try:
        logger.info(f"Predicting bugs for code (length: {len(request.code)})")
        
        # Extract features from code
        features = ml_service.extract_features(request.code)
        
        # Predict bug probability
        probability = ml_service.predict(features)
        
        # Determine severity
        severity = ml_service.get_severity(probability)
        
        # Generate explanation
        explanation = ml_service.generate_explanation(
            features, probability, severity
        )
        
        # Calculate metrics
        metrics = ml_service.calculate_metrics(request.code)
        
        return CodeAnalysisResponse(
            probability=float(probability),
            severity=severity,
            explanation=explanation,
            metrics=metrics,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error in predict_bug: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing code: {str(e)}"
        )

@app.post("/student-chat", response_model=StudentChatResponse)
async def student_chat(request: StudentChatRequest):
    """
    Provide AI-powered explanations for students
    """
    try:
        logger.info(f"Processing student chat request")
        
        # Use Gemini API to generate explanation
        explanation = await ai_service.explain_for_student(
            request.message,
            request.context or ""
        )
        
        # If code is provided, also analyze it
        if request.code:
            features = ml_service.extract_features(request.code)
            probability = ml_service.predict(features)
            severity = ml_service.get_severity(probability)
            
            # Add bug analysis to explanation
            bug_analysis = f"\n\nBug Analysis:\nProbability: {probability:.1%}\nSeverity: {severity}"
            explanation = explanation + bug_analysis
        
        return StudentChatResponse(
            response=explanation,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error in student_chat: {str(e)}")
        
        # Fallback response
        fallback_response = (
            "I'm here to help you learn about programming bugs! "
            "I can explain concepts like memory leaks, null pointers, "
            "infinite loops, and more. Please ask me specific questions "
            "or share code you'd like me to analyze."
        )
        
        return StudentChatResponse(
            response=fallback_response,
            success=False,
            error=str(e)
        )

@app.post("/developer-fix", response_model=DeveloperFixResponse)
async def developer_fix(request: DeveloperFixRequest):
    """
    Get AI-powered code fixes and explanations
    """
    try:
        logger.info(f"Processing developer fix request")
        
        # Use DeepSeek API to generate fix
        fix_result = await ai_service.fix_code(
            request.code,
            request.language or "python",
            request.issue_description or ""
        )
        
        # Analyze original code
        features = ml_service.extract_features(request.code)
        probability = ml_service.predict(features)
        severity = ml_service.get_severity(probability)
        
        # Analyze fixed code
        fixed_features = ml_service.extract_features(fix_result.get("fixed_code", ""))
        fixed_probability = ml_service.predict(fixed_features)
        fixed_severity = ml_service.get_severity(fixed_probability)
        
        # Calculate improvement
        improvement = probability - fixed_probability
        
        return DeveloperFixResponse(
            fixed_code=fix_result.get("fixed_code", request.code),
            explanation=fix_result.get("explanation", "No explanation provided"),
            original_severity=severity,
            fixed_severity=fixed_severity,
            improvement=float(improvement),
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error in developer_fix: {str(e)}")
        
        # Generate simple fallback fix
        fallback_fix = generate_fallback_fix(request.code)
        fallback_explanation = (
            "Here's a basic fix for common issues. "
            "Added null checks and input validation."
        )
        
        return DeveloperFixResponse(
            fixed_code=fallback_fix,
            explanation=fallback_explanation,
            original_severity="medium",
            fixed_severity="low",
            improvement=0.3,
            success=False,
            error=str(e)
        )

def generate_fallback_fix(code: str) -> str:
    """Generate a simple fallback fix when AI service is unavailable"""
    lines = code.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Simple pattern matching for common issues
        if 'len(' in line and '/' in line:
            # Add null/empty check before division
            indent = len(line) - len(line.lstrip())
            check_line = ' ' * indent + "if len(data) > 0:  # Added check for empty list"
            fixed_lines.append(check_line)
            fixed_lines.append(line)
        elif '.length' in line and ('<' in line or '>' in line or '<=' in line or '>=' in line):
            # Fix potential off-by-one
            line = line.replace('<=', '<').replace('>=', '>')
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

@app.get("/metrics")
async def get_system_metrics():
    """Get system metrics and statistics"""
    return {
        "model_accuracy": ml_service.get_accuracy(),
        "api_usage": {
            "total_requests": ml_service.request_count,
            "successful_predictions": ml_service.success_count
        },
        "ai_service_status": ai_service.is_ready()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )