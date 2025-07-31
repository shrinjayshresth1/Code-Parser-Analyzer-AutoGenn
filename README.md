# AutoGen + GenAI Enhanced Code Analyzer

A comprehensive Python code analysis tool that combines structured analysis (AST, symbol tables, control flow graphs) with AI-powered insights using AutoGen agents and GenAI providers.

## 🚀 Features

### AI-Powered Analysis
- **AutoGen Agents**: Multi-agent conversation system for intelligent code analysis
- **Code Quality**: Best practices and improvement suggestions
- **Security Analysis**: Potential vulnerabilities detection
- **Performance Insights**: Optimization recommendations  
- **Documentation**: Automatic code documentation generation

### Structured Analysis Engine
- **AST Parsing**: Complete abstract syntax tree with line/column info
- **Symbol Tables**: Functions, variables, parameters with scope information
- **Control Flow Graphs**: Program flow visualization and analysis
- **Data Flow Analysis**: Variable definitions and usage patterns

### GenAI Provider Support
- **OpenRouter**: Access to GPT-4, Claude, Llama, and more models
- **Groq**: High-speed inference with Llama and Mixtral models
- **Flexible Architecture**: Easy to extend with new AI providers

## 📁 Project Structure

```
Code-Parser-Analyzer-AutoGenn/
├── src/
│   └── code_analyzer/           # Main package
│       ├── __init__.py         # Package exports
│       └── analyzer.py         # Core analyzer implementation
├── tests/
│   ├── __init__.py
│   └── test_analyzer.py        # Comprehensive test suite
├── docs/
│   ├── usage_guide.md          # Detailed usage documentation
│   └── API_SETUP.md            # API key setup instructions
├── config/
│   └── .env.template           # Environment variables template
├── main.py                     # Main entry point script
├── setup.py                    # Package setup and installation
├── requirements.txt            # Python dependencies
├── .env                        # Your API keys (git-ignored)
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install
```bash
# Clone the repository
git clone https://github.com/devsinghsolanki/Code-Parser-Analyzer-AutoGenn.git
cd Code-Parser-Analyzer-AutoGenn

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
copy config/.env.template .env
# Edit .env with your API keys
```

### Development Install
```bash
# Install in development mode
pip install -e .

# Run from anywhere
code-analyzer
```

## 🔑 API Setup

### Option 1: Environment Variables (Recommended)

1. **Copy the template:**
   ```bash
   copy config/.env.template .env
   ```

2. **Edit `.env` with your API keys:**
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
   GROQ_API_KEY=gsk_your-actual-groq-key-here
   DEFAULT_PROVIDER=openrouter
   DEFAULT_MODEL=anthropic/claude-3-haiku
   ```

### Option 2: Manual Entry
The analyzer will prompt for API keys if no environment variables are found.

### Getting API Keys

**OpenRouter (Recommended)**
- Visit: https://openrouter.ai
- Sign up and navigate to "Keys" section
- Create a new API key (starts with `sk-or-v1-`)
- Supports: GPT-4, Claude, Llama, and 100+ models

**Groq (Fast Inference)**  
- Visit: https://console.groq.com
- Sign up and go to API Keys section
- Create a new API key (starts with `gsk_`)
- Supports: Llama 3, Mixtral models with ultra-fast inference

## 🚀 Usage

### Quick Start

**Basic Test (No API Key Required)**
```bash
python tests/test_analyzer.py
```

**Interactive Mode**
```bash
python main.py
```

**As Installed Package**
```bash
code-analyzer
```

### Programmatic Usage

```python
from code_analyzer import CodeAnalyzer, OpenRouterProvider

# Basic analysis (structured only)
analyzer = CodeAnalyzer()
result = analyzer.analyze(source_code)

# AI-enhanced analysis
provider = OpenRouterProvider("your-api-key")
analyzer = CodeAnalyzer(provider)
result = analyzer.analyze(source_code)

# Using environment variables
analyzer = CodeAnalyzer()  # Auto-detects API keys from .env
result = analyzer.analyze(source_code)
print(result["ai_insights"]["summary"])
```

### Example Analysis

```python
source_code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''

result = analyzer.analyze(source_code)

# Structured analysis (always available)
print("Functions found:", len([s for s in result["symbol_table"] if s["type"] == "function"]))
print("CFG nodes:", len(result["control_flow_graph"]["nodes"]))

# AI insights (with API key)
if result["ai_insights"]:
    print("Summary:", result["ai_insights"]["summary"])
    print("Suggestions:", result["ai_insights"]["suggestions"])
```

## 📊 Output Format

The analyzer produces a comprehensive JSON object:

```json
{
  "ast": { /* Complete abstract syntax tree */ },
  "symbol_table": [ /* All symbols with types and locations */ ],
  "control_flow_graph": { /* Program flow nodes and edges */ },
  "data_flow_graph": { /* Variable usage patterns */ },
  "ai_insights": { /* AI-powered analysis (when enabled) */ },
  "metadata": { 
    "total_lines": 25,
    "total_symbols": 8,
    "total_cfg_nodes": 12,
    "ai_enhanced": true,
    "autogen_enabled": true
  }
}
```

## 🧪 Testing

```bash
# Run basic functionality test
python tests/test_analyzer.py

# Test with environment variables
echo "OPENROUTER_API_KEY=your-key" > .env
python main.py
```

## 🔧 Development

### Project Architecture

- **`src/code_analyzer/analyzer.py`**: Core analysis engine
- **AutoGen Integration**: Multi-agent conversation system
- **Provider System**: Pluggable AI provider architecture
- **Fallback Mechanisms**: Graceful degradation when AI unavailable

### Adding New Providers

```python
class CustomProvider(GenAIProvider):
    def generate_response(self, prompt: str, model: str = None) -> str:
        # Implement your provider logic
        pass
    
    def get_autogen_config(self, model: str = None) -> dict:
        # Return AutoGen-compatible config
        pass
```

### Error Handling
- ✅ Graceful fallback when AutoGen unavailable
- ✅ API key validation and error reporting
- ✅ Network error handling with retries
- ✅ Malformed code parsing with detailed error messages

## 🔒 Security

- ✅ API keys stored in `.env` (git-ignored)
- ✅ No hardcoded credentials in source code
- ✅ Environment variable validation
- ✅ Secure API communication over HTTPS

## 📈 Performance

- **Structured Analysis**: Fast AST parsing and graph construction
- **Caching**: Intelligent caching of analysis results
- **Concurrent Processing**: Multi-threaded analysis for large codebases
- **Memory Efficient**: Optimized data structures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is provided as-is for educational and development purposes.

## 🆘 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**AutoGen Installation Issues**
```bash
pip install --upgrade pyautogen
```

**API Key Issues**
- Verify `.env` file exists and has correct format
- Check API key validity on provider websites
- Ensure no extra spaces in environment variables

### Getting Help

1. Check the [Usage Guide](docs/usage_guide.md)
2. Review [API Setup Instructions](docs/API_SETUP.md)
3. Run the test suite to verify installation
4. Open an issue on GitHub for bugs or feature requests

## 🎯 Roadmap

- [ ] VS Code Extension integration
- [ ] Support for more programming languages
- [ ] Real-time code analysis
- [ ] Web-based interface
- [ ] Integration with popular IDEs
- [ ] Advanced security analysis features

---

**Made with ❤️ using AutoGen and AI-powered analysis**
