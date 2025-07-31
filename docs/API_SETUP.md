# Setting Up Your API Keys

## Quick Setup

1. **Copy the template:**
   ```bash
   copy .env.template .env
   ```

2. **Edit the .env file** with your API keys:
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
   GROQ_API_KEY=gsk_your-actual-groq-key-here
   DEFAULT_PROVIDER=openrouter
   DEFAULT_MODEL=anthropic/claude-3-haiku
   ```

3. **Run the analyzer:**
   ```bash
   python code_analyzer.py
   ```

## Getting API Keys

### OpenRouter (Recommended)
- Visit: https://openrouter.ai
- Sign up and navigate to "Keys" section
- Create a new API key
- Copy the key that starts with `sk-or-v1-`

### Groq (Fast Inference)  
- Visit: https://console.groq.com
- Sign up and go to API Keys section
- Create a new API key
- Copy the key that starts with `gsk_`

## Security Notes

- ✅ `.env` is in `.gitignore` - your keys won't be committed
- ✅ Use `.env.template` as a reference (safe to commit)
- ✅ Never share your actual API keys
- ✅ Keys are loaded automatically when you run the analyzer

## Available Models

### OpenRouter Models:
- `anthropic/claude-3-haiku` (fast, cheap)
- `anthropic/claude-3-sonnet` (balanced)  
- `openai/gpt-4o-mini` (OpenAI)
- `meta-llama/llama-3.1-8b-instruct` (Llama)

### Groq Models:
- `llama3-8b-8192` (default)
- `llama3-70b-8192` (larger model)
- `mixtral-8x7b-32768` (Mixtral)
