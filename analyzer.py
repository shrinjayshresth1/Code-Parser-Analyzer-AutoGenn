#!/usr/bin/env python3
import ast
import json
import os
import sys
from typing import Dict, Any, List, Set
from datetime import datetime
from collections import defaultdict

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
    """Complete code analysis with AST, Symbol Table, and Control Flow Graph."""
    try:
        tree = ast.parse(code)
        
        # 1. AST Analysis
        ast_info = analyze_ast(tree)
        
        # 2. Symbol Table
        symbol_table = build_symbol_table(tree)
        
        # 3. Control Flow Graph
        cfg = build_control_flow_graph(tree)
        
        # 4. Metrics
        metrics = {
            "total_lines": len(code.split('\n')),
            "function_count": len(ast_info["functions"]),
            "class_count": len(ast_info["classes"]),
            "variable_count": len([s for s in symbol_table if s["type"] == "variable"]),
            "import_count": len([s for s in symbol_table if s["type"] == "import"]),
            "cfg_nodes": len(cfg["nodes"]),
            "cfg_edges": sum(len(node["successors"]) for node in cfg["nodes"])
        }
        
        return {
            "ast": ast_info,
            "symbol_table": symbol_table,
            "control_flow_graph": cfg,
            "metrics": metrics
        }
    
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

def analyze_ast(tree: ast.AST) -> Dict[str, Any]:
    """Extract AST information."""
    functions = []
    classes = []
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "col": node.col_offset,
                "args": [arg.arg for arg in node.args.args],
                "returns": ast.unparse(node.returns) if node.returns else None,
                "decorators": [ast.unparse(dec) for dec in node.decorator_list]
            })
        elif isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "col": node.col_offset,
                "bases": [ast.unparse(base) for base in node.bases],
                "decorators": [ast.unparse(dec) for dec in node.decorator_list],
                "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            })
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "type": "import"
                    })
            else:  # ImportFrom
                for alias in node.names:
                    imports.append({
                        "name": f"{node.module}.{alias.name}" if node.module else alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "type": "from_import",
                        "module": node.module
                    })
    
    return {
        "functions": functions,
        "classes": classes,
        "imports": imports
    }

def build_symbol_table(tree: ast.AST) -> List[Dict[str, Any]]:
    """Build comprehensive symbol table."""
    symbols = []
    current_scope = "global"
    
    class SymbolVisitor(ast.NodeVisitor):
        def __init__(self):
            self.scope_stack = ["global"]
        
        def visit_FunctionDef(self, node):
            # Function definition
            symbols.append({
                "name": node.name,
                "type": "function",
                "line": node.lineno,
                "col": node.col_offset,
                "scope": self.scope_stack[-1],
                "args": [arg.arg for arg in node.args.args]
            })
            
            # Parameters
            for arg in node.args.args:
                symbols.append({
                    "name": arg.arg,
                    "type": "parameter",
                    "line": arg.lineno,
                    "col": arg.col_offset,
                    "scope": node.name
                })
            
            # Enter function scope
            self.scope_stack.append(node.name)
            self.generic_visit(node)
            self.scope_stack.pop()
        
        def visit_ClassDef(self, node):
            symbols.append({
                "name": node.name,
                "type": "class",
                "line": node.lineno,
                "col": node.col_offset,
                "scope": self.scope_stack[-1]
            })
            
            self.scope_stack.append(node.name)
            self.generic_visit(node)
            self.scope_stack.pop()
        
        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.append({
                        "name": target.id,
                        "type": "variable",
                        "line": node.lineno,
                        "col": node.col_offset,
                        "scope": self.scope_stack[-1]
                    })
            self.generic_visit(node)
        
        def visit_Import(self, node):
            for alias in node.names:
                symbols.append({
                    "name": alias.asname or alias.name,
                    "type": "import",
                    "line": node.lineno,
                    "col": node.col_offset,
                    "scope": self.scope_stack[-1],
                    "original_name": alias.name
                })
        
        def visit_ImportFrom(self, node):
            for alias in node.names:
                symbols.append({
                    "name": alias.asname or alias.name,
                    "type": "import",
                    "line": node.lineno,
                    "col": node.col_offset,
                    "scope": self.scope_stack[-1],
                    "original_name": f"{node.module}.{alias.name}" if node.module else alias.name,
                    "from_module": node.module
                })
    
    visitor = SymbolVisitor()
    visitor.visit(tree)
    return symbols

def build_control_flow_graph(tree: ast.AST) -> Dict[str, Any]:
    """Build Control Flow Graph."""
    nodes = []
    edges = []
    node_id = 0
    
    def create_node(node_type: str, line: int, code: str = "") -> int:
        nonlocal node_id
        nodes.append({
            "id": node_id,
            "type": node_type,
            "line": line,
            "code": code.strip(),
            "successors": [],
            "predecessors": []
        })
        node_id += 1
        return node_id - 1
    
    def add_edge(from_id: int, to_id: int):
        edges.append({"from": from_id, "to": to_id})
        nodes[from_id]["successors"].append(to_id)
        nodes[to_id]["predecessors"].append(from_id)
    
    class CFGBuilder(ast.NodeVisitor):
        def __init__(self):
            self.current_node = create_node("entry", 1, "Program Start")
            self.exit_node = None
        
        def visit_stmt_list(self, stmts):
            prev = self.current_node
            for stmt in stmts:
                stmt_node = create_node("statement", stmt.lineno, ast.unparse(stmt)[:50])
                if prev is not None:
                    add_edge(prev, stmt_node)
                prev = stmt_node
                
                if isinstance(stmt, ast.If):
                    self.visit_If(stmt, stmt_node)
                elif isinstance(stmt, (ast.For, ast.While)):
                    self.visit_Loop(stmt, stmt_node)
            
            return prev
        
        def visit_If(self, node, if_node):
            # True branch
            true_branch = self.visit_stmt_list(node.body) if node.body else if_node
            
            # False branch (else/elif)
            false_branch = if_node
            if node.orelse:
                false_branch = self.visit_stmt_list(node.orelse)
            
            # Merge point
            merge_node = create_node("merge", node.lineno, "If-merge")
            if true_branch:
                add_edge(true_branch, merge_node)
            if false_branch and false_branch != if_node:
                add_edge(false_branch, merge_node)
            elif false_branch == if_node:
                add_edge(if_node, merge_node)
            
            return merge_node
        
        def visit_Loop(self, node, loop_node):
            # Loop body
            body_end = self.visit_stmt_list(node.body) if node.body else loop_node
            
            # Back edge to loop condition
            if body_end:
                add_edge(body_end, loop_node)
            
            # Exit from loop
            exit_node = create_node("statement", node.lineno, "Loop-exit")
            add_edge(loop_node, exit_node)
            
            return exit_node
    
    builder = CFGBuilder()
    if isinstance(tree, ast.Module):
        last_node = builder.visit_stmt_list(tree.body)
        exit_node = create_node("exit", len(tree.body), "Program End")
        if last_node:
            add_edge(last_node, exit_node)
    
    return {
        "nodes": nodes,
        "edges": edges
    }

def save_to_json(code: str, analysis_result: Dict[str, Any], ai_result: str = None, filename: str = "analysis_results.json"):
    """Save analysis results to a JSON file."""
    try:
        # Create analysis entry
        analysis_entry = {
            "timestamp": datetime.now().isoformat(),
            "code": code,
            "complete_analysis": analysis_result,
            "ai_analysis": ai_result
        }
        
        # Load existing results or create new list
        results = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                results = []
        
        # Add new analysis
        results.append(analysis_entry)
        
        # Save updated results
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")
        
    except Exception as e:
        print(f"Failed to save results: {e}")

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
    # Check for output file argument
    output_file = "analysis_results.json"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    print("Simple Code Analyzer")
    print("=" * 40)
    print(f"Output will be saved to: {output_file}")
    
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
            
            # Complete analysis
            analysis_result = analyze_code_basic(code)
            print("\nComplete Analysis:")
            print(json.dumps(analysis_result, indent=2))
            
            # AI analysis if available
            ai_result = None
            if user_proxy and analyst:
                print("\nAI Analysis:")
                ai_result = analyze_with_autogen(code, user_proxy, analyst)
                print(ai_result)
            
            # Save results to JSON file
            save_to_json(code, analysis_result, ai_result, output_file)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
