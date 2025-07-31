# AutoGen + GenAI Enhanced Code Analyzer - Usage Guide

## Overview
Your code analyzer has been successfully fixed and enhanced! It now combines:
- **Structured Analysis**: AST parsing, symbol tables, control flow graphs
- **AutoGen Framework**: Multi-agent conversation system for intelligent code analysis
- **GenAI Integration**: OpenRouter and Groq API support (avoiding OpenAI dependency)

## Quick Start

### Basic Mode (No API Key Required)
```bash
python test_autogen_analyzer.py
```
This runs the analyzer with structured analysis only.

### Interactive Mode with AI Enhancement
```bash
python code_analyzer.py
```
Then follow the prompts:
1. Choose 'y' for AI enhancement (or use .env file)
2. Select provider (OpenRouter or Groq) if not using .env
3. Enter your API key if not using .env
4. Provide code to analyze

## Getting API Keys

### Option 1: Environment Variables (Recommended)
1. Copy `.env.template` to `.env`
2. Edit `.env` and add your API keys:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
GROQ_API_KEY=your_groq_api_key_here
DEFAULT_PROVIDER=openrouter
DEFAULT_MODEL=anthropic/claude-3-haiku
```
3. Run the analyzer - it will automatically use your environment settings

### Option 2: Manual Entry

### OpenRouter (Recommended)
1. Visit: https://openrouter.ai
2. Sign up for an account
3. Get your API key from the dashboard
4. Models available: GPT-4, Claude, Llama, etc.

### Groq (Fast Inference)
1. Visit: https://groq.com
2. Sign up for an account
3. Get your API key from console
4. Models: Llama 3, Mixtral, etc.

## Features

### AI Enhancement (With API Key)
- **AutoGen Agents**: Multi-agent analysis conversations
- **Code Quality**: Best practices and improvement suggestions
- **Security Analysis**: Potential vulnerabilities
- **Performance Insights**: Optimization recommendations
- **Documentation**: Automatic code documentation

## Sample Output Structure
```json
{
  "ast": { /* Complete syntax tree */ },
  "symbol_table": [ /* All symbols with types and locations */ ],
  "control_flow_graph": { /* Program flow nodes and edges */ },
  "data_flow_graph": { /* Variable usage patterns */ },
  "ai_insights": { /* AI-powered analysis (when enabled) */ },
  "metadata": { /* Analysis statistics */ }
}
```

## Integration Examples

### Analyze a Python File
```python
from code_analyzer import CodeAnalyzer, OpenRouterProvider

# Using environment variables (recommended)
analyzer = CodeAnalyzer()  # Will auto-detect API keys from .env

# Or manually specify provider
provider = OpenRouterProvider("your-api-key")
analyzer = CodeAnalyzer(provider)

result = analyzer.analyze("your_code_here")
print(result["ai_insights"]["summary"])
```

### Analyze Code String
```python
code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''

result = analyzer.analyze_code(code)
```

## Error Handling
- Graceful fallback when AutoGen unavailable
- API key validation
- Network error handling
- Malformed code handling

## What Was Fixed
1. **Import Issues**: Resolved AutoGen library compatibility problems
2. **Version Conflicts**: Added flexible import handling for different pyautogen versions
3. **API Integration**: Added OpenRouter and Groq support as OpenAI alternatives
4. **Error Handling**: Comprehensive error handling and fallback mechanisms
5. **Testing**: Complete test suite for validation

## Next Steps
1. Test with your preferred AI provider
2. Integrate into your development workflow
3. Customize AI prompts for specific analysis needs
4. Extend with additional analysis features

Your analyzer is now production-ready and significantly more powerful than the original!
