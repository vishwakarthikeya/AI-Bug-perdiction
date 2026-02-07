"""
Utility functions for code analysis and feature extraction
"""

import re
import math
from typing import Dict, List, Tuple, Any
import ast
import tokenize
from io import StringIO

def extract_features_from_code(code: str) -> Dict[str, float]:
    """
    Extract various features from source code for ML prediction
    
    Args:
        code: Source code string
        
    Returns:
        Dictionary of feature names and values
    """
    features = {}
    
    try:
        # Basic metrics
        features['loc'] = count_lines_of_code(code)
        features['avg_line_length'] = calculate_average_line_length(code)
        features['comment_density'] = calculate_comment_density(code)
        
        # Complexity metrics
        features['cyclomatic_complexity'] = calculate_cyclomatic_complexity(code)
        features['halstead_volume'] = calculate_halstead_volume(code)
        
        # Structural features
        features['num_functions'] = count_functions(code)
        features['num_classes'] = count_classes(code)
        features['num_loops'] = count_loops(code)
        features['num_conditionals'] = count_conditionals(code)
        features['num_try_except'] = count_try_except(code)
        features['num_imports'] = count_imports(code)
        
        # Bug pattern indicators
        features['num_null_checks'] = count_null_checks(code)
        features['num_div_operations'] = count_division_operations(code)
        features['num_array_access'] = count_array_access(code)
        
        # Code quality indicators
        features['nested_depth'] = calculate_max_nesting_depth(code)
        features['parameter_count'] = calculate_avg_parameters(code)
        
        # Language-specific features (Python)
        if is_python_code(code):
            python_features = extract_python_specific_features(code)
            features.update(python_features)
        
        # Ensure all features are finite numbers
        for key in features:
            if not isinstance(features[key], (int, float)) or not math.isfinite(features[key]):
                features[key] = 0.0
        
    except Exception as e:
        # If feature extraction fails, return minimal features
        features = get_minimal_features(code)
    
    return features

def count_lines_of_code(code: str) -> int:
    """Count non-empty lines of code"""
    lines = code.strip().split('\n')
    return sum(1 for line in lines if line.strip())

def calculate_average_line_length(code: str) -> float:
    """Calculate average line length"""
    lines = code.strip().split('\n')
    if not lines:
        return 0
    total_length = sum(len(line) for line in lines)
    return total_length / len(lines)

def calculate_comment_density(code: str) -> float:
    """Calculate comment density (percentage of comment lines)"""
    lines = code.strip().split('\n')
    if not lines:
        return 0
    
    comment_lines = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
            comment_lines += 1
    
    return (comment_lines / len(lines)) * 100

def calculate_cyclomatic_complexity(code: str) -> float:
    """
    Calculate cyclomatic complexity (simplified)
    Counts decision points in code
    """
    complexity = 1  # Start with 1
    
    # Count decision points
    patterns = [
        r'\bif\s*\(', r'\belse\s+if\s*\(', r'\belse\b',
        r'\bfor\s*\(', r'\bwhile\s*\(', r'\bdo\b',
        r'\bcase\b', r'\bdefault\b',
        r'\b&&\b', r'\b\|\|\b', r'\bcatch\b',
        r'\b\?\s*:', r'\bthrow\b'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        complexity += len(matches)
    
    return float(complexity)

def calculate_halstead_volume(code: str) -> float:
    """
    Calculate Halstead volume (simplified)
    Measures program complexity based on operators and operands
    """
    # Simple token-based implementation
    operators = set()
    operands = set()
    
    # Common operators
    operator_patterns = [
        r'\+', r'-', r'\*', r'/', r'%', r'=', r'==', r'!=', r'<', r'>', r'<=', r'>=',
        r'\+=', r'-=', r'\*=', r'/=', r'%=',
        r'\+\+', r'--',
        r'&&', r'\|\|', r'!',
        r'&', r'\|', r'~', r'^', r'<<', r'>>',
        r'\.', r'->', r'::'
    ]
    
    # Count unique operators
    for pattern in operator_patterns:
        if re.search(pattern, code):
            operators.add(pattern)
    
    # Simple operand counting (identifiers and literals)
    identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
    identifiers = re.findall(identifier_pattern, code)
    operands.update(identifiers)
    
    literal_pattern = r'\b\d+(\.\d+)?\b|\b["\'][^"\']*["\']\b'
    literals = re.findall(literal_pattern, code)
    operands.update(literals)
    
    n1 = len(operators)  # Number of distinct operators
    n2 = len(operands)   # Number of distinct operands
    N1 = sum(len(re.findall(op, code)) for op in operators)  # Total operators
    N2 = sum(len(re.findall(re.escape(op), code)) for op in operands)  # Total operands
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    # Halstead volume formula
    N = N1 + N2
    n = n1 + n2
    volume = N * math.log2(n) if n > 0 else 0
    
    return float(volume)

def count_functions(code: str) -> int:
    """Count function/method definitions"""
    patterns = [
        r'\bdef\s+\w+\s*\(',  # Python
        r'\bfunction\s+\w+\s*\(',  # JavaScript
        r'\bpublic|private|protected\s+\w+\s+\w+\s*\(',  # Java/C#
        r'\bvoid\s+\w+\s*\(',  # C/C++
        r'\bint\s+\w+\s*\(', r'\bfloat\s+\w+\s*\(', r'\bdouble\s+\w+\s*\(',
        r'\bchar\s+\w+\s*\(', r'\bbool\s+\w+\s*\('
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        count += len(matches)
    
    return count

def count_classes(code: str) -> int:
    """Count class definitions"""
    patterns = [
        r'\bclass\s+\w+',  # Python, Java, C++, etc.
        r'\binterface\s+\w+',  # Java
        r'\bstruct\s+\w+',  # C/C++
        r'\benum\s+\w+'  # Various languages
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        count += len(matches)
    
    return count

def count_loops(code: str) -> int:
    """Count loop statements"""
    patterns = [
        r'\bfor\s*\(', r'\bwhile\s*\(', r'\bdo\s*\{',
        r'\bfor\s+\w+\s+in\b',  # Python for-in
        r'\bforeach\s*\(',  # C#
        r'\bfor\s*\(\s*\w+\s*:\s*\w+\s*\)'  # Java enhanced for
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        count += len(matches)
    
    return count

def count_conditionals(code: str) -> int:
    """Count conditional statements"""
    patterns = [
        r'\bif\s*\(', r'\belse\s+if\s*\(', r'\bswitch\s*\(',
        r'\bcase\s+\w+:', r'\bdefault:',
        r'\belif\s*\(',  # Python
        r'\bunless\s*\(',  # Ruby
        r'\bguard\s+'  # Swift
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        count += len(matches)
    
    return count

def count_try_except(code: str) -> int:
    """Count try-catch/except blocks"""
    patterns = [
        r'\btry\s*\{', r'\bcatch\s*\(', r'\bfinally\s*\{',
        r'\btry:', r'\bexcept\b',  # Python
        r'\brescue\b'  # Ruby
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        count += len(matches)
    
    return count

def count_imports(code: str) -> int:
    """Count import/include statements"""
    patterns = [
        r'^\s*import\s+', r'^\s*from\s+\w+\s+import',  # Python
        r'^\s*#include\s+',  # C/C++
        r'^\s*using\s+',  # C#
        r'^\s*require\s+', r'^\s*import\s+',  # JavaScript
        r'^\s*package\s+',  # Java
        r'^\s*@import\s+'  # CSS
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, code, re.MULTILINE | re.IGNORECASE)
        count += len(matches)
    
    return count

def count_null_checks(code: str) -> int:
    """Count null/None checks"""
    patterns = [
        r'!=?\s*null\b', r'==?\s*null\b',
        r'!=?\s*None\b', r'==?\s*None\b',
        r'!=?\s*undefined\b', r'==?\s*undefined\b',
        r'!=?\s*NULL\b', r'==?\s*NULL\b',
        r'\bif\s*\(\s*\w+\s*\)',  # if(variable) - common null check pattern
        r'\boptional\b', r'\bOptional\b'  # Optional types
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        count += len(matches)
    
    return count

def count_division_operations(code: str) -> int:
    """Count division operations (potential division by zero)"""
    patterns = [
        r'/\s*\w+', r'/\s*\d+',
        r'%=\s*\w+', r'%=\s*\d+',
        r'//\s*\w+', r'//\s*\d+'  # Python integer division
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, code)
        count += len(matches)
    
    return count

def count_array_access(code: str) -> int:
    """Count array/list access operations"""
    patterns = [
        r'\w+\s*\[',  # array[index]
        r'\.get\s*\(', r'\.set\s*\(',  # Java get/set
        r'\.at\s*\(',  # C++ at()
        r'\.slice\s*\(', r'\.splice\s*\(',  # JavaScript
        r'\.append\s*\(', r'\.pop\s*\(', r'\.insert\s*\(\)'  # Python list operations
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, code)
        count += len(matches)
    
    return count

def calculate_max_nesting_depth(code: str) -> int:
    """Calculate maximum nesting depth"""
    lines = code.split('\n')
    max_depth = 0
    current_depth = 0
    
    for line in lines:
        stripped = line.strip()
        # Count opening braces/brackets
        opens = stripped.count('{') + stripped.count('[') + stripped.count('(')
        # Count closing braces/brackets
        closes = stripped.count('}') + stripped.count(']') + stripped.count(')')
        
        current_depth += opens - closes
        max_depth = max(max_depth, current_depth)
        
        # Reset if negative (shouldn't happen in valid code)
        if current_depth < 0:
            current_depth = 0
    
    return max_depth

def calculate_avg_parameters(code: str) -> float:
    """Calculate average number of parameters per function"""
    # Find function definitions and count parameters
    function_patterns = [
        r'def\s+\w+\s*\((.*?)\)',  # Python
        r'function\s+\w+\s*\((.*?)\)',  # JavaScript
        r'\w+\s+\w+\s*\((.*?)\)\s*\{'  # C-style languages
    ]
    
    total_params = 0
    function_count = 0
    
    for pattern in function_patterns:
        matches = re.findall(pattern, code, re.DOTALL)
        for params in matches:
            # Count parameters (split by comma, ignoring whitespace)
            param_list = [p.strip() for p in params.split(',') if p.strip()]
            total_params += len(param_list)
            function_count += 1
    
    return total_params / function_count if function_count > 0 else 0.0

def is_python_code(code: str) -> bool:
    """Check if code appears to be Python"""
    python_indicators = [
        r'^import\s+\w+',
        r'^from\s+\w+\s+import',
        r'def\s+\w+\s*\(.*\)\s*:',
        r'class\s+\w+\s*:',
        r'print\s*\(.*\)',
        r'if\s+.*:',
        r'for\s+.*:',
        r'while\s+.*:'
    ]
    
    for pattern in python_indicators:
        if re.search(pattern, code, re.MULTILINE):
            return True
    
    return False

def extract_python_specific_features(code: str) -> Dict[str, float]:
    """Extract Python-specific features"""
    features = {}
    
    try:
        # Parse Python code
        tree = ast.parse(code)
        
        # Count various AST nodes
        features['num_ast_nodes'] = len(list(ast.walk(tree)))
        
        # Count specific node types
        features['num_list_comprehensions'] = len([
            node for node in ast.walk(tree) 
            if isinstance(node, ast.ListComp)
        ])
        
        features['num_dict_comprehensions'] = len([
            node for node in ast.walk(tree) 
            if isinstance(node, ast.DictComp)
        ])
        
        features['num_generator_expressions'] = len([
            node for node in ast.walk(tree) 
            if isinstance(node, ast.GeneratorExp)
        ])
        
        features['num_lambda_expressions'] = len([
            node for node in ast.walk(tree) 
            if isinstance(node, ast.Lambda)
        ])
        
        features['num_decorators'] = len([
            node for node in ast.walk(tree) 
            if isinstance(node, ast.FunctionDef) and node.decorator_list
        ])
        
        # Check for common Python patterns
        code_lower = code.lower()
        features['has_assert'] = int('assert ' in code_lower)
        features['has_raise'] = int('raise ' in code_lower)
        features['has_yield'] = int('yield ' in code_lower)
        features['has_async'] = int('async ' in code_lower)
        features['has_await'] = int('await ' in code_lower)
        
    except SyntaxError:
        # Not valid Python code
        pass
    
    return features

def get_minimal_features(code: str) -> Dict[str, float]:
    """Get minimal features when full extraction fails"""
    return {
        'loc': count_lines_of_code(code),
        'avg_line_length': calculate_average_line_length(code),
        'comment_density': calculate_comment_density(code),
        'cyclomatic_complexity': 1.0,
        'halstead_volume': 0.0,
        'num_functions': 0,
        'num_classes': 0,
        'num_loops': 0,
        'num_conditionals': 0,
        'num_try_except': 0,
        'num_imports': 0,
        'num_null_checks': 0,
        'num_div_operations': 0,
        'num_array_access': 0,
        'nested_depth': 0,
        'parameter_count': 0.0
    }

def validate_code_syntax(code: str, language: str = "python") -> Tuple[bool, str]:
    """
    Validate code syntax
    
    Args:
        code: Source code to validate
        language: Programming language
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if language.lower() == "python":
        try:
            ast.parse(code)
            return True, "Syntax is valid"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
    
    # For other languages, do basic validation
    lines = code.split('\n')
    
    # Check for unbalanced braces/brackets/parentheses
    stack = []
    pairs = {'{': '}', '[': ']', '(': ')'}
    
    for line_num, line in enumerate(lines, 1):
        for char in line:
            if char in pairs:
                stack.append((char, line_num))
            elif char in pairs.values():
                if not stack:
                    return False, f"Unmatched '{char}' on line {line_num}"
                opening, opening_line = stack.pop()
                if pairs[opening] != char:
                    return False, f"Mismatched '{opening}' on line {opening_line} with '{char}' on line {line_num}"
    
    if stack:
        char, line_num = stack.pop()
        return False, f"Unclosed '{char}' on line {line_num}"
    
    return True, "Basic syntax appears valid"

def calculate_code_similarity(code1: str, code2: str) -> float:
    """
    Calculate similarity between two code snippets (simplified)
    
    Returns:
        Similarity score between 0 and 1
    """
    # Tokenize and compare
    tokens1 = re.findall(r'\b\w+\b', code1.lower())
    tokens2 = re.findall(r'\b\w+\b', code2.lower())
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0