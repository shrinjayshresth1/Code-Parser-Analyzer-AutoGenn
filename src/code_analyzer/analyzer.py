"""
AutoGen + GenAI Enhanced Code Parser and Analyzer - Converts Python code to structured IR
Requirements:
1. Take a string of source code as input
2. Parse it to create an Abstract Syntax Tree (AST)
3. Analyze the AST to build a symbol table
4. Analyze the AST to generate a control flow graph (CFG)
5. Output all information as a single, language-neutral JSON object
6. Enhanced with AutoGen agents using OpenRouter or Groq
"""

import ast
import json
import requests
import os
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file if it exists."""
    # Check for .env in current directory first, then config directory
    env_paths = [
        os.path.join(os.getcwd(), '.env'),
        os.path.join(os.path.dirname(__file__), '..', '..', '.env'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'config', '.env')
    ]
    
    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            break

# Load .env file at module import
load_env_file()

# AutoGen imports - try different possible import paths
try:
    from autogen import AssistantAgent, UserProxyAgent
except ImportError:
    try:
        from autogen.agentchat import AssistantAgent, UserProxyAgent
    except ImportError:
        try:
            from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
        except ImportError:
            print("Warning: AutoGen not available. Installing...")
            import subprocess
            subprocess.run(["pip", "install", "pyautogen"], check=True)
            from autogen import AssistantAgent, UserProxyAgent


@dataclass
class SymbolInfo:
    """Information about a symbol in the code."""
    name: str
    type: str  # 'function', 'class', 'variable', 'parameter', 'import'
    line: int
    col: int
    scope: str
    value: Optional[str] = None
    ai_description: Optional[str] = None


@dataclass
class CFGNode:
    """A node in the Control Flow Graph."""
    id: int
    type: str  # 'entry', 'exit', 'statement', 'branch', 'loop'
    line: int
    code: str
    successors: List[int]
    predecessors: List[int]
    ai_analysis: Optional[str] = None


class GenAIProvider:
    """Base class for GenAI providers."""
    
    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url
    
    def generate_response(self, prompt: str, model: str = None) -> str:
        """Generate response from AI provider."""
        raise NotImplementedError


class OpenRouterProvider(GenAIProvider):
    """OpenRouter API provider compatible with AutoGen."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://openrouter.ai/api/v1")
    
    def generate_response(self, prompt: str, model: str = "anthropic/claude-3-haiku") -> str:
        """Generate response using OpenRouter API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/code-analyzer",
                "X-Title": "AutoGen Code Analyzer"
            }
            
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"OpenRouter API Error: {str(e)}"
    
    def get_autogen_config(self, model: str = "anthropic/claude-3-haiku") -> dict:
        """Get AutoGen-compatible configuration."""
        return {
            "model": model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_type": "openai"  # OpenRouter is OpenAI-compatible
        }


class GroqProvider(GenAIProvider):
    """Groq API provider compatible with AutoGen."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.groq.com/openai/v1")
    
    def generate_response(self, prompt: str, model: str = "llama3-8b-8192") -> str:
        """Generate response using Groq API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Groq API Error: {str(e)}"
    
    def get_autogen_config(self, model: str = "llama3-8b-8192") -> dict:
        """Get AutoGen-compatible configuration."""
        return {
            "model": model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_type": "openai"  # Groq is OpenAI-compatible
        }


class AutoGenCodeAnalysisAgent:
    """AutoGen-based code analysis agent system."""
    
    def __init__(self, provider: GenAIProvider, model: str = None):
        self.provider = provider
        self.model = model
        
        # Create AutoGen config
        if hasattr(provider, 'get_autogen_config'):
            config = provider.get_autogen_config(model) if model else provider.get_autogen_config()
            self.config_list = [config]
        else:
            # Fallback for direct API usage
            self.config_list = None
        
        self._setup_agents()
    
    def _setup_agents(self):
        """Set up AutoGen agents for code analysis."""
        try:
            # If AutoGen is available, use it
            if 'autogen' in globals() or any('autogen' in str(type(obj)) for obj in globals().values()):
                self.user_proxy = UserProxyAgent(
                    name="User",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=3,
                    code_execution_config=False,
                    system_message="You are a user requesting code analysis."
                )
                
                self.analyst = AssistantAgent(
                    name="CodeAnalyst",
                    system_message="""You are a code analysis expert. When given Python code, analyze it and return ONLY a JSON object with the following structure:
{
    "summary": "Brief description of what the code does",
    "complexity_analysis": "Time and space complexity analysis",
    "suggestions": ["List of code quality suggestions"],
    "patterns_detected": ["List of design patterns found"],
    "potential_issues": ["List of potential problems or improvements"]
}

Always return valid JSON without any additional text or markdown formatting.""",
                    llm_config={"config_list": self.config_list} if self.config_list else None
                )
                
                self.autogen_available = True
            else:
                self.autogen_available = False
                
        except Exception as e:
            print(f"AutoGen setup failed: {e}")
            self.autogen_available = False
    
    def analyze_with_autogen(self, source_code: str) -> dict:
        """Use AutoGen agents to analyze code."""
        if not self.autogen_available:
            # Fallback to direct API call
            return self._direct_analysis(source_code)
        
        try:
            # Use AutoGen conversation
            message = f"""Analyze this Python code and return insights as JSON:

```python
{source_code}
```

Return ONLY the JSON object without any additional text."""
            
            # Initiate chat between agents
            self.user_proxy.initiate_chat(
                self.analyst,
                message=message,
                max_turns=1
            )
            
            # Get the last message from the analyst
            chat_history = self.user_proxy.chat_messages.get(self.analyst, [])
            if chat_history:
                last_message = chat_history[-1].get('content', '')
                return self._parse_ai_response(last_message)
            else:
                return self._direct_analysis(source_code)
                
        except Exception as e:
            print(f"AutoGen analysis failed: {e}")
            return self._direct_analysis(source_code)
    
    def _direct_analysis(self, source_code: str) -> dict:
        """Direct API call fallback when AutoGen is not available."""
        prompt = f"""
Analyze this Python code and provide insights in JSON format:

```python
{source_code}
```

Please provide:
1. A brief summary of what the code does
2. Complexity analysis (time/space complexity if applicable)
3. Code quality suggestions
4. Design patterns or programming patterns detected
5. Potential issues or improvements

Return response as valid JSON with keys: summary, complexity_analysis, suggestions, patterns_detected, potential_issues
"""
        
        ai_response = self.provider.generate_response(prompt)
        return self._parse_ai_response(ai_response)
    
    def _parse_ai_response(self, ai_response: str) -> dict:
        """Parse AI response and extract JSON."""
        try:
            # Extract JSON from response if it's wrapped in markdown
            if "```json" in ai_response:
                json_start = ai_response.find("```json") + 7
                json_end = ai_response.find("```", json_start)
                ai_response = ai_response[json_start:json_end].strip()
            elif "```" in ai_response:
                json_start = ai_response.find("```") + 3
                json_end = ai_response.rfind("```")
                ai_response = ai_response[json_start:json_end].strip()
            
            return json.loads(ai_response)
        except json.JSONDecodeError:
            return {
                "summary": ai_response,
                "complexity_analysis": "Could not parse structured response",
                "suggestions": [],
                "patterns_detected": [],
                "potential_issues": []
            }


class CodeAnalyzer:
    """AutoGen-enhanced code analyzer that combines structured analysis with AI insights."""
    
    def __init__(self, ai_provider: GenAIProvider = None, model: str = None):
        """Initialize analyzer with optional GenAI provider and AutoGen agents."""
        self.ai_provider = ai_provider
        self.model = model
        self.symbols: List[SymbolInfo] = []
        self.cfg_nodes: List[CFGNode] = []
        self.current_scope = "global"
        self.node_counter = 0
        
        # Set up AutoGen agent if provider is available
        self.autogen_agent = None
        if ai_provider:
            self.autogen_agent = AutoGenCodeAnalysisAgent(ai_provider, model)
        
    def analyze(self, source_code: str) -> Dict[str, Any]:
        """
        Main analysis function that converts source code to structured IR with GenAI enhancement.
        
        Args:
            source_code: String containing Python source code
            
        Returns:
            Dictionary containing AST, symbol table, CFG, and AI insights as JSON-serializable data
        """
        try:
            # Step 1: Parse source code to create AST
            tree = ast.parse(source_code)
            
            # Step 2: Convert AST to JSON-serializable format
            ast_json = self._ast_to_dict(tree)
            
            # Step 3: Build symbol table by traversing AST
            self._build_symbol_table(tree)
            
            # Step 4: Generate Control Flow Graph
            self._build_cfg(tree)
            
            # Step 5: Generate Data Flow information
            data_flow = self._analyze_data_flow(tree)
            
            # Step 6: AutoGen AI Enhancement (if provider available)
            ai_insights = None
            if self.autogen_agent:
                ai_insights = self.autogen_agent.analyze_with_autogen(source_code)
                self._enhance_symbols_with_ai(source_code)
                self._enhance_cfg_with_ai(source_code)
            
            # Return complete IR as JSON object
            return {
                "ast": ast_json,
                "symbol_table": [asdict(symbol) for symbol in self.symbols],
                "control_flow_graph": {
                    "nodes": [asdict(node) for node in self.cfg_nodes],
                    "entry_node": 0 if self.cfg_nodes else None,
                    "exit_nodes": self._find_exit_nodes()
                },
                "data_flow_graph": data_flow,
                "ai_insights": ai_insights,
                "metadata": {
                    "total_lines": len(source_code.split('\n')),
                    "total_symbols": len(self.symbols),
                    "total_cfg_nodes": len(self.cfg_nodes),
                    "ai_enhanced": bool(self.autogen_agent),
                    "autogen_enabled": bool(self.autogen_agent and self.autogen_agent.autogen_available)
                }
            }
            
        except SyntaxError as e:
            return {
                "error": "Syntax Error",
                "message": str(e),
                "line": e.lineno,
                "offset": e.offset
            }
        except Exception as e:
            return {
                "error": "Analysis Error", 
                "message": str(e)
            }
    
    def _ast_to_dict(self, node: ast.AST) -> Dict[str, Any]:
        """Convert AST node to JSON-serializable dictionary."""
        result = {
            "type": node.__class__.__name__,
            "line": getattr(node, 'lineno', None),
            "col": getattr(node, 'col_offset', None)
        }
        
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                result[field] = [self._ast_to_dict(item) if isinstance(item, ast.AST) 
                               else item for item in value]
            elif isinstance(value, ast.AST):
                result[field] = self._ast_to_dict(value)
            else:
                result[field] = value
                
        return result
    
    def _build_symbol_table(self, tree: ast.AST):
        """Build symbol table by traversing the AST."""
        class SymbolVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.scope_stack = ["global"]
            
            def visit_FunctionDef(self, node):
                # Add function to symbol table
                self.analyzer.symbols.append(SymbolInfo(
                    name=node.name,
                    type="function",
                    line=node.lineno,
                    col=node.col_offset,
                    scope=self.scope_stack[-1]
                ))
                
                # Enter function scope
                self.scope_stack.append(node.name)
                
                # Add parameters
                for arg in node.args.args:
                    self.analyzer.symbols.append(SymbolInfo(
                        name=arg.arg,
                        type="parameter",
                        line=arg.lineno if hasattr(arg, 'lineno') else node.lineno,
                        col=arg.col_offset if hasattr(arg, 'col_offset') else 0,
                        scope=node.name
                    ))
                
                self.generic_visit(node)
                self.scope_stack.pop()
            
            def visit_ClassDef(self, node):
                self.analyzer.symbols.append(SymbolInfo(
                    name=node.name,
                    type="class",
                    line=node.lineno,
                    col=node.col_offset,
                    scope=self.scope_stack[-1]
                ))
                
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()
            
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.analyzer.symbols.append(SymbolInfo(
                            name=target.id,
                            type="variable",
                            line=node.lineno,
                            col=node.col_offset,
                            scope=self.scope_stack[-1]
                        ))
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    self.analyzer.symbols.append(SymbolInfo(
                        name=name,
                        type="import",
                        line=node.lineno,
                        col=node.col_offset,
                        scope=self.scope_stack[-1],
                        value=alias.name
                    ))
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    self.analyzer.symbols.append(SymbolInfo(
                        name=name,
                        type="import",
                        line=node.lineno,
                        col=node.col_offset,
                        scope=self.scope_stack[-1],
                        value=f"{node.module}.{alias.name}" if node.module else alias.name
                    ))
                self.generic_visit(node)
        
        visitor = SymbolVisitor(self)
        visitor.visit(tree)
    
    def _build_cfg(self, tree: ast.AST):
        """Build Control Flow Graph from AST."""
        # Create entry node
        entry_node = CFGNode(
            id=self._next_node_id(),
            type="entry",
            line=1,
            code="<entry>",
            successors=[],
            predecessors=[]
        )
        self.cfg_nodes.append(entry_node)
        
        # Process the AST to build CFG
        last_node_id = self._process_statements(tree.body, entry_node.id)
        
        # Create exit node
        exit_node = CFGNode(
            id=self._next_node_id(),
            type="exit",
            line=getattr(tree, 'end_lineno', 0) or 0,
            code="<exit>",
            successors=[],
            predecessors=[last_node_id] if isinstance(last_node_id, int) else last_node_id
        )
        self.cfg_nodes.append(exit_node)
        
        # Connect last statements to exit
        if isinstance(last_node_id, int):
            self._add_edge(last_node_id, exit_node.id)
        else:
            for node_id in last_node_id:
                self._add_edge(node_id, exit_node.id)
    
    def _process_statements(self, stmts: List[ast.stmt], prev_node_id: int) -> int:
        """Process a list of statements and return the last node ID."""
        current_node_id = prev_node_id
        
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                current_node_id = self._process_if(stmt, current_node_id)
            elif isinstance(stmt, ast.While):
                current_node_id = self._process_while(stmt, current_node_id)
            elif isinstance(stmt, ast.For):
                current_node_id = self._process_for(stmt, current_node_id)
            else:
                # Regular statement
                node = CFGNode(
                    id=self._next_node_id(),
                    type="statement",
                    line=getattr(stmt, 'lineno', 0),
                    code=ast.unparse(stmt) if hasattr(ast, 'unparse') else str(type(stmt).__name__),
                    successors=[],
                    predecessors=[current_node_id]
                )
                self.cfg_nodes.append(node)
                self._add_edge(current_node_id, node.id)
                current_node_id = node.id
        
        return current_node_id
    
    def _process_if(self, if_stmt: ast.If, prev_node_id: int) -> int:
        """Process if statement and return merge node ID."""
        # Branch node
        branch_node = CFGNode(
            id=self._next_node_id(),
            type="branch",
            line=if_stmt.lineno,
            code=f"if {ast.unparse(if_stmt.test) if hasattr(ast, 'unparse') else 'condition'}",
            successors=[],
            predecessors=[prev_node_id]
        )
        self.cfg_nodes.append(branch_node)
        self._add_edge(prev_node_id, branch_node.id)
        
        # Process if body
        then_last = self._process_statements(if_stmt.body, branch_node.id)
        
        # Process else body
        else_last = branch_node.id
        if if_stmt.orelse:
            else_last = self._process_statements(if_stmt.orelse, branch_node.id)
        
        # Create merge node
        merge_node = CFGNode(
            id=self._next_node_id(),
            type="statement",
            line=if_stmt.lineno,
            code="<merge>",
            successors=[],
            predecessors=[]
        )
        self.cfg_nodes.append(merge_node)
        
        # Connect branches to merge
        if isinstance(then_last, int):
            self._add_edge(then_last, merge_node.id)
        if isinstance(else_last, int):
            self._add_edge(else_last, merge_node.id)
        
        return merge_node.id
    
    def _process_while(self, while_stmt: ast.While, prev_node_id: int) -> int:
        """Process while loop and return exit node ID."""
        # Loop header
        loop_node = CFGNode(
            id=self._next_node_id(),
            type="loop",
            line=while_stmt.lineno,
            code=f"while {ast.unparse(while_stmt.test) if hasattr(ast, 'unparse') else 'condition'}",
            successors=[],
            predecessors=[prev_node_id]
        )
        self.cfg_nodes.append(loop_node)
        self._add_edge(prev_node_id, loop_node.id)
        
        # Process loop body
        body_last = self._process_statements(while_stmt.body, loop_node.id)
        
        # Connect body back to loop header
        if isinstance(body_last, int):
            self._add_edge(body_last, loop_node.id)
        
        return loop_node.id
    
    def _process_for(self, for_stmt: ast.For, prev_node_id: int) -> int:
        """Process for loop and return exit node ID."""
        # Similar to while loop
        loop_node = CFGNode(
            id=self._next_node_id(),
            type="loop",
            line=for_stmt.lineno,
            code=f"for {ast.unparse(for_stmt.target) if hasattr(ast, 'unparse') else 'var'} in {ast.unparse(for_stmt.iter) if hasattr(ast, 'unparse') else 'iterable'}",
            successors=[],
            predecessors=[prev_node_id]
        )
        self.cfg_nodes.append(loop_node)
        self._add_edge(prev_node_id, loop_node.id)
        
        body_last = self._process_statements(for_stmt.body, loop_node.id)
        if isinstance(body_last, int):
            self._add_edge(body_last, loop_node.id)
        
        return loop_node.id
    
    def _analyze_data_flow(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze data flow in the code."""
        variables = defaultdict(list)
        definitions = defaultdict(list)
        uses = defaultdict(list)
        
        class DataFlowVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        definitions[target.id].append({
                            "line": node.lineno,
                            "col": node.col_offset,
                            "type": "definition"
                        })
                self.generic_visit(node)
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    uses[node.id].append({
                        "line": node.lineno,
                        "col": node.col_offset,
                        "type": "use"
                    })
                self.generic_visit(node)
        
        visitor = DataFlowVisitor()
        visitor.visit(tree)
        
        return {
            "definitions": dict(definitions),
            "uses": dict(uses),
            "variables": list(set(list(definitions.keys()) + list(uses.keys())))
        }
    
    def _next_node_id(self) -> int:
        """Get next available node ID."""
        node_id = self.node_counter
        self.node_counter += 1
        return node_id
    
    def _add_edge(self, from_id: int, to_id: int):
        """Add edge between CFG nodes."""
        # Find nodes and update successors/predecessors
        from_node = next((n for n in self.cfg_nodes if n.id == from_id), None)
        to_node = next((n for n in self.cfg_nodes if n.id == to_id), None)
        
        if from_node and to_node:
            if to_id not in from_node.successors:
                from_node.successors.append(to_id)
            if from_id not in to_node.predecessors:
                to_node.predecessors.append(from_id)
    
    def _find_exit_nodes(self) -> List[int]:
        """Find all exit nodes in the CFG."""
        return [node.id for node in self.cfg_nodes if node.type == "exit"]
    
    def _get_ai_insights(self, source_code: str) -> Dict[str, Any]:
        """Get AI-powered insights about the code."""
        if not self.ai_provider:
            return None
        
        try:
            prompt = f"""
Analyze this Python code and provide insights in JSON format:

```python
{source_code}
```

Please provide:
1. A brief summary of what the code does
2. Complexity analysis (time/space complexity if applicable)
3. Code quality suggestions
4. Design patterns or programming patterns detected
5. Potential issues or improvements

Return response as valid JSON with keys: summary, complexity_analysis, suggestions, patterns_detected, potential_issues
"""
            
            ai_response = self.ai_provider.generate_response(prompt)
            
            # Try to parse as JSON, fallback to structured response
            try:
                # Extract JSON from response if it's wrapped in markdown
                if "```json" in ai_response:
                    json_start = ai_response.find("```json") + 7
                    json_end = ai_response.find("```", json_start)
                    ai_response = ai_response[json_start:json_end].strip()
                elif "```" in ai_response:
                    json_start = ai_response.find("```") + 3
                    json_end = ai_response.rfind("```")
                    ai_response = ai_response[json_start:json_end].strip()
                
                return json.loads(ai_response)
            except json.JSONDecodeError:
                return {
                    "summary": ai_response,
                    "complexity_analysis": "Could not parse structured response",
                    "suggestions": [],
                    "patterns_detected": [],
                    "potential_issues": []
                }
                
        except Exception as e:
            return {
                "summary": f"AI analysis failed: {str(e)}",
                "complexity_analysis": None,
                "suggestions": [],
                "patterns_detected": []
            }
    
    def _enhance_symbols_with_ai(self, source_code: str):
        """Enhance symbol table with AI descriptions."""
        if not self.ai_provider:
            return
        
        try:
            # Group symbols by type for batch processing
            functions = [s for s in self.symbols if s.type == "function"]
            classes = [s for s in self.symbols if s.type == "class"]
            
            # Get AI descriptions for functions and classes
            for symbol in functions + classes:
                symbol.ai_description = self._get_symbol_description(source_code, symbol)
                
        except Exception:
            pass  # Continue without AI enhancement if it fails
    
    def _get_symbol_description(self, source_code: str, symbol: SymbolInfo) -> str:
        """Get AI description for a specific symbol."""
        try:
            prompt = f"""
In this Python code:
```python
{source_code}
```

Describe the {symbol.type} '{symbol.name}' in one concise sentence (max 50 words).
Focus on its purpose and functionality.
"""
            
            response = self.ai_provider.generate_response(prompt)
            return response.strip()
            
        except Exception:
            return None
    
    def _enhance_cfg_with_ai(self, source_code: str):
        """Enhance CFG nodes with AI analysis."""
        if not self.ai_provider:
            return
        
        try:
            # Enhance branch and loop nodes with AI analysis
            for node in self.cfg_nodes:
                if node.type in ["branch", "loop"] and not node.code.startswith("<"):
                    node.ai_analysis = self._get_node_analysis(source_code, node)
                    
        except Exception:
            pass  # Continue without AI enhancement if it fails
    
    def _get_node_analysis(self, source_code: str, node: CFGNode) -> str:
        """Get AI analysis for a specific CFG node."""
        try:
            prompt = f"""
In this Python code:
```python
{source_code}
```

Analyze this {node.type} statement: "{node.code}"
Provide a brief analysis of its purpose and behavior in one sentence.
"""
            
            response = self.ai_provider.generate_response(prompt)
            return response.strip()
            
        except Exception:
            return None


def analyze_code(source_code: str, provider: str = None, api_key: str = None, model: str = None) -> str:
    """
    Main function to analyze source code and return JSON IR with AutoGen + GenAI enhancement.
    
    Args:
        source_code: String containing Python source code
        provider: AI provider ('openrouter', 'groq', or None for no AI)
        api_key: API key for the chosen provider
        model: Model to use (optional, uses default if not specified)
        
    Returns:
        JSON string containing structured intermediate representation with AutoGen insights
    """
    ai_provider = None
    
    if provider and api_key:
        if provider.lower() == 'openrouter':
            ai_provider = OpenRouterProvider(api_key)
        elif provider.lower() == 'groq':
            ai_provider = GroqProvider(api_key)
        else:
            print(f"Warning: Unknown provider '{provider}'. Running without AI enhancement.")
    
    analyzer = CodeAnalyzer(ai_provider, model)
    result = analyzer.analyze(source_code)
    return json.dumps(result, indent=2)


def main():
    """Interactive AutoGen + GenAI enhanced code analyzer."""
    print("AutoGen + GenAI Enhanced Code Parser and Analyzer")
    print("Combines structured analysis (AST + Symbol Table + CFG) with AutoGen AI agents")
    print("Supported providers: OpenRouter, Groq")
    print("="*75)
    
    # Check for environment variables
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY')
    default_provider = os.getenv('DEFAULT_PROVIDER', 'openrouter').lower()
    default_model = os.getenv('DEFAULT_MODEL')
    
    # Get AI provider configuration
    provider = None
    api_key = None
    model = None
    
    if openrouter_key or groq_key:
        print(f"\n✓ Found API keys in environment:")
        if openrouter_key:
            print("  - OpenRouter API key loaded")
        if groq_key:
            print("  - Groq API key loaded")
        
        use_env = input(f"\nUse environment settings? (y/n, default: y): ").strip().lower()
        if use_env != 'n':
            if default_provider == 'groq' and groq_key:
                provider = 'groq'
                api_key = groq_key
            elif openrouter_key:
                provider = 'openrouter'
                api_key = openrouter_key
            elif groq_key:
                provider = 'groq'
                api_key = groq_key
            
            model = default_model
            print(f"✓ Using {provider.upper()} from environment")
            if model:
                print(f"✓ Using model: {model}")
    
    if not provider:
        use_ai = input("Use AutoGen + GenAI enhancement? (y/n): ").strip().lower()
        
        if use_ai == 'y':
            print("\nAvailable providers:")
            print("1. OpenRouter (supports multiple models including Claude, GPT, Llama, etc.)")
            print("2. Groq (fast inference with Llama models)")
            
            provider_choice = input("Choose provider (1 for OpenRouter, 2 for Groq): ").strip()
            
            if provider_choice == '1':
                provider = 'openrouter'
                api_key = input("Enter OpenRouter API key: ").strip()
                print("\nPopular models:")
                print("- anthropic/claude-3-haiku (default, fast and cheap)")
                print("- anthropic/claude-3-sonnet (balanced)")
                print("- openai/gpt-4o-mini (OpenAI)")
                print("- meta-llama/llama-3.1-8b-instruct (Llama)")
                model = input("Enter model (press Enter for default): ").strip() or None
            elif provider_choice == '2':
                provider = 'groq'
                api_key = input("Enter Groq API key: ").strip()
                print("\nAvailable models:")
                print("- llama3-8b-8192 (default)")
                print("- llama3-70b-8192")
                print("- mixtral-8x7b-32768")
                model = input("Enter model (press Enter for default): ").strip() or None
            else:
                print("Invalid choice. Running without AI enhancement.")
            
            if provider and api_key:
                print(f"✓ Using {provider.upper()} with AutoGen agents")
                if model:
                    print(f"✓ Model: {model}")
            else:
                print("✓ Running in basic mode (no AI enhancement)")
        else:
            print("✓ Running in basic mode (no AI enhancement)")
    
    print("\nType 'quit' to exit\n")
    
    while True:
        print("Enter Python code (press Enter twice to analyze):")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "" and lines:
                    break
                if line.strip().lower() == 'quit':
                    print("Goodbye!")
                    return
                lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return
        
        if not lines:
            continue
            
        source_code = '\n'.join(lines)
        
        print("\n" + "="*75)
        print("AUTOGEN + AI ENHANCED ANALYSIS RESULT:")
        print("="*75)
        
        result = analyze_code(source_code, provider, api_key, model)
        print(result)
        print("\n" + "="*75 + "\n")


if __name__ == "__main__":
    main()
