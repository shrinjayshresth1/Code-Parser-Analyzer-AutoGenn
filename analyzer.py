#!/usr/bin/env python3
"""
Simple Code Analyzer with AutoGen
A minimal implementation for code analysis using AST and AutoGen agents.
"""

import ast
import json
import os
from typing import Dict, Any

# Load environment variables
def load_env():
    """Load API key from .env file or environment."""
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

# Try to import AutoGen
try:
    from autogen import AssistantAgent, UserProxyAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    print("AutoGen not available. Install with: pip install pyautogen")
    AUTOGEN_AVAILABLE = False

def analyze_code_basic(code: str) -> Dict[str, Any]:
    """Basic code analysis using Python AST."""
    try:
        tree = ast.parse(code)
        
        # Extract basic information
        functions = []
        classes = []
        variables = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args]
                })
            elif isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "line": node.lineno
                })
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append({
                            "name": target.id,
                            "line": node.lineno
                        })
        
        return {
            "ast_summary": {
                "functions": functions,
                "classes": classes,
                "variables": variables
            },
            "metrics": {
                "total_lines": len(code.split('\n')),
                "function_count": len(functions),
                "class_count": len(classes),
                "variable_count": len(variables)
            }
        }
    
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

def setup_autogen():
    """Setup AutoGen agents if available."""
    if not AUTOGEN_AVAILABLE:
        return None, None
    
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('API_KEY')
    if not api_key:
        print("Warning: No API key found. Set OPENAI_API_KEY in .env file")
        return None, None
    
    config_list = [{
        "model": "gpt-3.5-turbo",
        "api_key": api_key
    }]
    
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False
    )
    
    analyst = AssistantAgent(
        name="CodeAnalyst",
        system_message="You are a code analyst. Analyze the given code and return a JSON summary with AST, symbols, and insights.",
        llm_config={"config_list": config_list}
    )
    
    return user_proxy, analyst

def analyze_with_autogen(code: str, user_proxy, analyst) -> str:
    """Analyze code using AutoGen agents."""
    try:
        response = user_proxy.initiate_chat(
            analyst,
            message=f"Analyze this Python code and return JSON analysis:\n```python\n{code}\n```",
            silent=True
        )
        return str(response)
    except Exception as e:
        return f"AutoGen analysis failed: {e}"

def main():
    """Main interactive analyzer."""
    print("Simple Code Analyzer")
    print("=" * 40)
    
    # Setup AutoGen if available
    user_proxy, analyst = setup_autogen()
    if user_proxy and analyst:
        print("AutoGen enabled with AI analysis")
    else:
        print("Basic AST analysis only")
    
    print("\nEnter Python code to analyze (type 'quit' to exit):")
    
    while True:
        try:
            print("\n" + "-" * 40)
            code = input(">>> ")
            
            if code.lower() in ['quit', 'exit', 'q']:
                break
            
            if not code.strip():
                continue
            
            # Basic analysis
            basic_result = analyze_code_basic(code)
            print("\nBasic Analysis:")
            print(json.dumps(basic_result, indent=2))
            
            # AI analysis if available
            if user_proxy and analyst:
                print("\nAI Analysis:")
                ai_result = analyze_with_autogen(code, user_proxy, analyst)
                print(ai_result)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
