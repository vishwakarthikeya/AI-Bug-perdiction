"""
Minimal schemas for AI Bug Predictor
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from enum import Enum

class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class CodeMetrics(BaseModel):
    loc: int = 0
    complexity: float = 0.0
    num_functions: int = 0
    num_loops: int = 0
    num_conditionals: int = 0
    issues: int = 0
    maintainability: float = 0.0

class CodeAnalysisRequest(BaseModel):
    code: str
    language: Optional[str] = "python"
    context: Optional[str] = None

class CodeAnalysisResponse(BaseModel):
    probability: float = 0.0
    severity: SeverityLevel = SeverityLevel.LOW
    explanation: str = ""
    metrics: CodeMetrics = CodeMetrics()
    success: bool = True
    error: Optional[str] = None

class StudentChatRequest(BaseModel):
    message: str
    code: Optional[str] = None
    context: Optional[str] = None
    difficulty: Optional[str] = "beginner"

class StudentChatResponse(BaseModel):
    response: str = ""
    success: bool = True
    error: Optional[str] = None
    suggestions: Optional[List[str]] = None

class DeveloperFixRequest(BaseModel):
    code: str
    language: str = "python"
    issue_description: Optional[str] = None
    focus_areas: Optional[List[str]] = None

class DeveloperFixResponse(BaseModel):
    fixed_code: str = ""
    explanation: str = ""
    original_severity: SeverityLevel = SeverityLevel.LOW
    fixed_severity: SeverityLevel = SeverityLevel.LOW
    improvement: float = 0.0
    success: bool = True
    error: Optional[str] = None
    changes: Optional[List[str]] = None