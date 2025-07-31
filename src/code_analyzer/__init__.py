"""
AutoGen + GenAI Enhanced Code Parser and Analyzer

This package provides comprehensive Python code analysis capabilities including:
- Abstract Syntax Tree (AST) parsing
- Symbol table construction  
- Control Flow Graph (CFG) generation
- Data flow analysis
- AI-powered insights via AutoGen agents
- Support for OpenRouter and Groq API providers

Author: GitHub Copilot
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GitHub Copilot"

from .analyzer import CodeAnalyzer, analyze_code
from .analyzer import OpenRouterProvider, GroqProvider, AutoGenCodeAnalysisAgent

__all__ = [
    "CodeAnalyzer",
    "analyze_code", 
    "OpenRouterProvider",
    "GroqProvider",
    "AutoGenCodeAnalysisAgent"
]
