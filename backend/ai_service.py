"""
AI Service for Bug Predictor - Simplified Working Version
"""

import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AIService:
    """Simplified AI service with pre-defined responses"""
    
    def __init__(self):
        self.is_initialized = True
        logger.info("AI Service initialized successfully")
    
    async def explain_for_student(self, question: str, context: str = "") -> str:
        """
        Provide educational explanations for students
        
        Args:
            question: Student's question
            context: Additional context or code
            
        Returns:
            Educational explanation
        """
        try:
            logger.info(f"Student question: {question[:50]}...")
            
            # Simulate AI processing time
            await asyncio.sleep(0.3)
            
            # Generate response based on keywords
            response = self._generate_student_response(question, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in explain_for_student: {e}")
            return self._get_fallback_explanation()
    
    async def fix_code(self, code: str, language: str = "python", issue_description: str = "") -> Dict[str, Any]:
        """
        Provide code fixes
        
        Args:
            code: Source code to fix
            language: Programming language
            issue_description: Optional issue description
            
        Returns:
            Dictionary with fixed code and explanation
        """
        try:
            logger.info(f"Fixing {language} code")
            
            # Simulate processing
            await asyncio.sleep(0.5)
            
            # Generate fix based on code patterns
            result = self._generate_code_fix(code, language, issue_description)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fix_code: {e}")
            return self._get_fallback_fix(code, language)
    
    def _generate_student_response(self, question: str, context: str) -> str:
        """Generate response based on keywords in question"""
        question_lower = question.lower()
        
        # Memory leaks
        if any(word in question_lower for word in ['memory', 'leak', 'malloc', 'free']):
            return ""

