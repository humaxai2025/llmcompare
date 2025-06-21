# 🚀 LLM Comparator

A powerful, feature-rich CLI tool to compare responses from multiple Large Language Models (LLMs) and score them for quality. Get the best answers by testing your prompts across OpenAI, Anthropic, Google, and local Ollama models simultaneously.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

## ✨ Features

### 🎯 **Multi-Provider Support**
- **OpenAI**: GPT-4o, GPT-4o-mini, and other models
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Haiku, Claude 3 Opus
- **Google**: Gemini 2.0 Flash (Experimental), Gemini 1.5 Flash, Gemini 1.5 Pro
- **Ollama**: Local models (Llama, Mistral, and more)

### 🧠 **Intelligent Scoring System**
- **LLM Judge**: Uses GPT-4o as an intelligent judge to score responses
- **Multi-Criteria Evaluation**: Relevance, clarity, completeness, accuracy, creativity, helpfulness
- **Weighted Scoring**: Customizable scoring weights for different criteria
- **Winner Selection**: Automatically identifies the best response

### 🎨 **Rich CLI Experience**
- **Beautiful Tables**: Rich-formatted output with colors and styling
- **Progress Bars**: Real-time progress tracking for multiple API calls
- **Interactive Mode**: Chat-like interface for iterative comparisons
- **Multiple Output Formats**: Table, JSON, CSV, Markdown

### 📝 **Smart Prompt Templates**
- **Technical**: Detailed explanations for developers
- **Simple**: Beginner-friendly explanations
- **Creative**: Engaging, story-driven responses
- **Academic**: Scholarly analysis with formal language
- **Business**: ROI-focused, practical applications
- **Code**: Programming tasks with examples
- **Debug**: Systematic troubleshooting help
- **Review**: Constructive feedback and suggestions

### 📊 **Advanced Features**
- **Cost Tracking**: Real-time API cost calculation
- **Response Time Monitoring**: Performance metrics for each model
- **History Management**: SQLite database for comparison tracking
- **Search & Replay**: Find and replay previous comparisons
- **Configuration Management**: YAML-based settings with smart defaults

## 🛠 Installation

### Quick Install (Recommended)

1. **Download the installer:**
   ```bash
   # Download setup.py and llm_comparator.py to a folder
   ```

2. **Run the installer:**
   ```bash
   python setup.py
   ```

3. **Follow the installation prompts** - it will:
   - Create a virtual environment
   - Install all dependencies
   - Set up configuration files
   - Create launcher scripts

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install aiohttp openai anthropic google-generativeai rich pyyaml python-dotenv

# Download the main script
# Place llmcompare.py in your chosen directory
```

## ⚙️ Configuration

### 1. Set Up API Keys

Copy the `.env.template` to `.env` and add your API keys:

```bash
cp .env.template .env
```

Edit `.env` file:
```env
# Get your API keys from:
OPENAI_API_KEY=sk-your-actual-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here  
GOOGLE_API_KEY=your-actual-google-key-here

# Optional: Local Ollama settings
OLLAMA_HOST=http://localhost:11434
```

### 2. Get API Keys

| Provider | Get API Key | Free Tier |
|----------|-------------|-----------|
| **OpenAI** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | $5 free credit |
| **Anthropic** | [console.anthropic.com](https://console.anthropic.com/) | $5 free credit |
| **Google** | [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey) | Free tier available |
| **Ollama** | Local models | Free (run locally) |

> **💡 Pro Tip**: You only need **ONE** API key to start using the tool!

### 3. Set Up Ollama (Optional)

For local models:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Install models
ollama pull llama3.2
ollama pull mistral
ollama pull codellama
```

## 🚀 Quick Start

> **💡 Execution Options:** You can run the tool in two ways:
> - **Direct Python:** `python llmcompare.py [options]` (works everywhere)
> - **Convenience Launchers:** `llm-compare` or `./llm-compare` (if installed via setup.py)

### Basic Usage

```bash
# Simple comparison
python llmcompare.py "Explain quantum computing"

# Check your setup
python llmcompare.py --config

# Interactive mode
python llmcompare.py --interactive
```

### Example Output

**Basic Comparison:**
```bash
$ python llmcompare.py "Explain quantum computing in simple terms"

🔥 LLM Comparison Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model                    ┃ Score ┃ Time   ┃ Cost     ┃ Response Preview               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ anthropic-claude-3-5-so… │  8.7  │ 3.2s   │ $0.0156  │ Quantum computing is like...   │
│ openai-gpt-4o 🏆         │  8.9  │ 2.8s   │ $0.0142  │ Think of quantum computing...  │
│ google-gemini-1.5-flash  │  8.1  │ 2.1s   │ $0.0008  │ Quantum computers work by...   │
└──────────────────────────┴───────┴────────┴──────────┴────────────────────────────────┘

Winner: openai-gpt-4o
```

**Configuration Check:**
```bash
$ python llmcompare.py --config

🔧 LLM Comparator Configuration

API Keys Status:
  ✓ OpenAI (GPT models): Configured
  ✓ Anthropic (Claude models): Configured  
  ✓ Google (Gemini models): Configured

.env file:
  ✓ Found at: /home/user/llm-comparator/.env

Available Providers:
  ✓ openai
  ✓ anthropic
  ✓ google
```

### Using Templates

```bash
# Technical explanation
python llmcompare.py --template technical --topic "machine learning"

# Business analysis
python llmcompare.py --template business --topic "cloud migration"

# Code generation
python llmcompare.py --template code --language "Python" --task "web scraper"
```

### Example Template Output

**Technical Template:**
```bash
$ python llmcompare.py --template technical --topic "Docker containers"

Generated prompt: Explain Docker containers to a software engineer with technical details and code examples where appropriate.

🔥 LLM Comparison Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model                    ┃ Score ┃ Time   ┃ Cost     ┃ Response Preview               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ anthropic-claude-3-5-so… │  9.1  │ 4.1s   │ $0.0234  │ Docker containers are lightw...│
│ openai-gpt-4o 🏆         │  9.3  │ 3.5s   │ $0.0198  │ ## Docker Containers: A Tech...│
│ google-gemini-1.5-flash  │  8.8  │ 2.3s   │ $0.0012  │ Docker containers package app..│
└──────────────────────────┴───────┴────────┴──────────┴────────────────────────────────┘
```

## 📖 Command Reference

### 🎯 Basic Commands

```bash
# Simple comparison
python llmcompare.py "your prompt here"

# Interactive mode
python llmcompare.py --interactive
python llmcompare.py -i
```

### 📋 Templates

| Template | Usage | Required Parameters |
|----------|-------|-------------------|
| `technical` | Technical explanations | `--topic` |
| `simple` | Beginner-friendly | `--topic` |
| `creative` | Engaging narratives | `--topic` |
| `academic` | Scholarly analysis | `--topic` |
| `business` | Business perspective | `--topic` |
| `comparison` | Compare concepts | `--topic` |
| `howto` | Step-by-step guides | `--action`, `--topic` |
| `debug` | Troubleshooting help | `--problem` |
| `code` | Programming tasks | `--language`, `--task` |
| `review` | Code/content review | `--type`, `--content` |

**Examples:**
```bash
python llmcompare.py --template technical --topic "Docker containers"
python llmcompare.py --template code --language "Python" --task "sort algorithm"
python llmcompare.py --template debug --problem "React component not rendering"
```

### 🎛 Model Selection

```bash
# Specific providers
python llmcompare.py --providers openai,anthropic "test prompt"
python llmcompare.py --providers google,ollama "test prompt"

# Custom models (JSON)
python llmcompare.py --models '{"openai": ["gpt-4o"], "anthropic": ["claude-3-5-sonnet-latest"]}' "prompt"
```

### 📊 Output Formats

```bash
# Table (default) - beautiful rich tables
python llmcompare.py "prompt"

# JSON - for data processing
python llmcompare.py --format json "prompt" > results.json

# CSV - for spreadsheet analysis  
python llmcompare.py --format csv "prompt" > results.csv

# Markdown - for documentation
python llmcompare.py --format markdown "prompt" > results.md

# Detailed scoring breakdown
python llmcompare.py --detailed "prompt"
```

### Example Output Formats

**JSON Format:**
```bash
$ python llmcompare.py --format json "What is AI?" 

{
  "id": "comp_1672531200",
  "prompt": "What is AI?",
  "timestamp": "2025-01-21T15:30:45",
  "winner": "openai-gpt-4o",
  "responses": [
    {
      "provider": "openai",
      "model": "gpt-4o",
      "response": "Artificial Intelligence (AI) refers to...",
      "response_time": 2.8,
      "token_count": 156,
      "cost": 0.0142,
      "error": null
    }
  ],
  "scores": {
    "openai-gpt-4o": {
      "relevance": 0.9,
      "clarity": 0.85,
      "completeness": 0.88
    }
  }
}
```

**CSV Format:**
```bash
$ python llmcompare.py --format csv "What is AI?"

model,provider,score,time,cost,tokens,error
gpt-4o,openai,8.900,2.80,0.0142,156,
claude-3-5-sonnet-latest,anthropic,8.700,3.20,0.0156,148,
gemini-1.5-flash,google,8.100,2.10,0.0008,142,
```

**Detailed Scoring:**
```bash
$ python llmcompare.py --detailed "What is AI?"

🔥 LLM Comparison Results
[Main results table]

┌─ openai-gpt-4o ─────────────────────┐
│ relevance       9.0 ██████████      │
│ clarity         8.5 ████████▓░      │
│ completeness    8.8 ████████▓▓      │
│ accuracy        9.2 █████████▓      │
│ creativity      7.5 ███████▓░░      │
│ helpfulness     8.9 ████████▓▓      │
│ OVERALL         8.9 ████████▓▓      │
└─────────────────────────────────────┘

┌─ anthropic-claude-3-5-sonnet-latest ┐
│ relevance       8.8 ████████▓▓      │
│ clarity         9.1 █████████▓      │
│ completeness    8.5 ████████▓░      │
│ accuracy        8.9 ████████▓▓      │
│ creativity      8.2 ████████▓░      │
│ helpfulness     8.7 ████████▓▓      │
│ OVERALL         8.7 ████████▓▓      │
└─────────────────────────────────────┘
```

### 📚 History & Search

```bash
# Show recent comparisons
python llmcompare.py --history

# Search history
python llmcompare.py --history --search "quantum"

# Load specific comparison
python llmcompare.py --load comp_1672531200
```

### Example History Output

**Recent History:**
```bash
$ python llmcompare.py --history

Comparison History
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID       ┃ Prompt                                                       ┃ Winner                   ┃ Time                     ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ comp_125 │ Explain machine learning in simple terms                    │ openai-gpt-4o            │ 2025-01-21 15:30:45     ┃
│ comp_124 │ What is quantum computing                                    │ anthropic-claude-3-5-s...│ 2025-01-21 15:25:12     ┃
│ comp_123 │ Debug React component not rendering                         │ openai-gpt-4o            │ 2025-01-21 15:20:38     ┃
│ comp_122 │ Write Python code to sort a list                           │ anthropic-claude-3-5-s...│ 2025-01-21 15:15:22     ┃
└──────────┴──────────────────────────────────────────────────────────────┴──────────────────────────┴─────────────────────────┘
```

**Search History:**
```bash
$ python llmcompare.py --history --search "quantum"

Comparison History
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID       ┃ Prompt                                                       ┃ Winner                   ┃ Time                     ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ comp_124 │ What is quantum computing                                    │ anthropic-claude-3-5-s...│ 2025-01-21 15:25:12     ┃
│ comp_118 │ Explain quantum physics for beginners                       │ openai-gpt-4o            │ 2025-01-21 14:45:33     ┃
└──────────┴──────────────────────────────────────────────────────────────┴──────────────────────────┴─────────────────────────┘
```

### ⚙️ Configuration

```bash
# Show config and API key status
python llmcompare.py --config

# Show available templates
python llmcompare.py --templates

# Setup help for API keys
python llmcompare.py --setup-help
```

### Example Configuration Outputs

**Templates List:**
```bash
$ python llmcompare.py --templates

technical: Explain {topic} to a software engineer with technical details and code examples where appropriate.
simple: Explain {topic} in simple terms that a beginner can understand.
creative: Write a creative and engaging explanation of {topic} using analogies and storytelling.
academic: Provide a scholarly analysis of {topic} with citations and formal language.
business: Explain {topic} from a business perspective, focusing on practical applications and ROI.
comparison: Compare and contrast different aspects of {topic}, highlighting pros and cons.
howto: Provide step-by-step instructions on how to {action} related to {topic}.
debug: Help debug this issue: {problem}. Provide systematic troubleshooting steps.
code: Write {language} code to {task}. Include comments and error handling.
review: Review this {type}: {content}. Provide constructive feedback and suggestions.
```

**Setup Help:**
```bash
$ python llmcompare.py --setup-help

🔑 API Key Setup Instructions

Step 1: Create .env file
Create a new file: /home/user/llm-comparator/.env

Step 2: Add your API keys to .env file
Edit the .env file and add:
OPENAI_API_KEY=sk-your-actual-openai-key
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key
GOOGLE_API_KEY=your-actual-google-key

Step 3: Get API keys
• OpenAI: https://platform.openai.com/api-keys
• Anthropic: https://console.anthropic.com/
• Google: https://makersuite.google.com/app/apikey

Note: You only need ONE API key to start using the tool!
```

### 🐛 Debug & Verbose

```bash
# Verbose output
python llmcompare.py --verbose "prompt"
python llmcompare.py -v "prompt"

# Debug mode (detailed logging)
python llmcompare.py --debug "prompt"

# Combined debug options
python llmcompare.py --verbose --debug --detailed "prompt"
```

### Example Debug Output

**Verbose Mode:**
```bash
$ python llmcompare.py --verbose "What is AI?"

INFO: Loaded .env file from /home/user/llm-comparator/.env
INFO: Available providers: OpenAI (GPT models), Anthropic (Claude models), Google (Gemini models)
INFO: Initialized 3 API client(s): openai, anthropic, google
INFO: Querying 3 models...
INFO: Querying OpenAI gpt-4o
INFO: Querying Anthropic claude-3-5-sonnet-latest
INFO: Querying Google gemini-1.5-flash
INFO: Scoring responses...
INFO: Running LLM judge evaluation

🔥 LLM Comparison Results
[Results table displayed]
```

**Debug Mode:**
```bash
$ python llmcompare.py --debug "What is AI?"

DEBUG: Found valid OPENAI_API_KEY
DEBUG: Found valid ANTHROPIC_API_KEY
DEBUG: Found valid GOOGLE_API_KEY
DEBUG: OpenAI client initialized successfully
DEBUG: Anthropic client initialized successfully
DEBUG: Google client initialized successfully
INFO: Querying 3 models...
DEBUG: Anthropic response usage: Usage(input_tokens=12, output_tokens=148)
DEBUG: Input tokens: 12, Output tokens: 148
DEBUG: Model for cost calc: claude-3-5-sonnet-latest
DEBUG: Calculated cost: $0.002556
DEBUG: Google tokens: input=12, output=142
DEBUG: Cost calc: provider=google, model=gemini-1.5-flash
DEBUG: Available models in config: ['gemini-1.5-flash', 'gemini-1.5-pro']
DEBUG: Calculated cost: $0.000828

🔥 LLM Comparison Results
[Results table displayed]
```

## 🎮 Interactive Mode

Launch interactive mode for iterative experimentation:

```bash
python llmcompare.py --interactive
```

**Available commands:**
- `compare` - Run a new comparison
- `history [search_term]` - Show/search comparison history  
- `config` - Show current configuration
- `templates` - List available templates
- `help` - Show help
- `quit` - Exit

### Example Interactive Session

```bash
$ python llmcompare.py --interactive

🚀 LLM Comparator Interactive Mode
Commands: compare, history, config, templates, help, quit

llm-compare> compare
Enter your prompt: What is machine learning?
Use a template? (y/n) [n]: y
Available templates: technical, simple, creative, academic, business, comparison, howto, debug, code, review
Template name: simple
Enter topic: machine learning
Generated prompt: Explain machine learning in simple terms that a beginner can understand.

[Comparison runs...]

🔥 LLM Comparison Results
[Results table displayed]

llm-compare> history
Recent Comparisons:
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID       ┃ Prompt                         ┃ Winner                   ┃ Time                     ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ comp_123 │ Explain machine learning...    │ openai-gpt-4o            │ 2025-01-21 15:30:45     ┃
│ comp_122 │ What is quantum computing...   │ anthropic-claude-3-5-s...│ 2025-01-21 15:25:12     ┃
└──────────┴────────────────────────────────┴──────────────────────────┴─────────────────────────┘

llm-compare> quit
Goodbye!
```

## 🧪 Testing

### Run Unit Tests

The project includes a comprehensive test suite that covers all functionality:

```bash
# Run all tests
python test_llmcompare.py

# Run specific test categories
python -m unittest test_llmcompare.TestConfigManager -v
python -m unittest test_llmcompare.TestHistoryManager -v
python -m unittest test_llmcompare.TestOutputFormatter -v

# Run with debug output
python test_llmcompare.py --verbose
```

### Test Coverage

✅ **Configuration Management** - Config loading, templates, API key validation  
✅ **History & Database** - SQLite operations, search functionality  
✅ **Output Formatting** - JSON, CSV, Markdown, table formats  
✅ **CLI Arguments** - All command-line options and flags  
✅ **API Integration** - Mocked tests for all providers  
✅ **Error Handling** - Edge cases and failure scenarios  

### Test Requirements

- All tests use mocks - **no real API keys needed**
- Tests run in isolated environments
- Automatic cleanup of temporary files
- Cross-platform compatibility

**Example test output:**
```bash
🧪 Starting LLM Comparator Unit Tests
==================================================
Tests run: 21
Failures: 0  
Errors: 0
Success rate: 100.0%
✅ All tests passed! 🎉
```

### Multi-Provider Technical Analysis
```bash
python llmcompare.py --template technical --topic "Kubernetes architecture" \
  --providers openai,anthropic,google --detailed --format markdown > k8s_analysis.md
```

### Example Advanced Output

**Multi-Model Comparison with Detailed Scoring:**
```bash
$ python llmcompare.py --template code --language "Python" --task "web scraper" --detailed

Generated prompt: Write Python code to web scraper. Include comments and error handling.

INFO: Querying 4 models...
INFO: Scoring responses...

🔥 LLM Comparison Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model                    ┃ Score ┃ Time   ┃ Cost     ┃ Response Preview               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ anthropic-claude-3-5-so… │  9.2  │ 4.1s   │ $0.0234  │ import requests from bs4...    │
│ openai-gpt-4o 🏆         │  9.4  │ 3.5s   │ $0.0198  │ import requests import time... │
│ google-gemini-1.5-flash  │  8.9  │ 2.3s   │ $0.0012  │ import requests from bs4...    │
│ ollama-llama3.2          │  8.1  │ 12.4s  │ $0.0000  │ Here's a Python web scraper... │
└──────────────────────────┴───────┴────────┴──────────┴────────────────────────────────┘

┌─ openai-gpt-4o ─────────────────────┐
│ relevance       9.5 █████████▓      │
│ clarity         9.2 █████████▓      │
│ completeness    9.6 █████████▓      │
│ accuracy        9.4 █████████▓      │
│ creativity      8.8 ████████▓▓      │
│ helpfulness     9.5 █████████▓      │
│ OVERALL         9.4 █████████▓      │
└─────────────────────────────────────┘

┌─ anthropic-claude-3-5-sonnet-latest ┐
│ relevance       9.3 █████████▓      │
│ clarity         9.0 █████████░      │
│ completeness    9.1 █████████▓      │
│ accuracy        9.2 █████████▓      │
│ creativity      9.0 █████████░      │
│ helpfulness     9.4 █████████▓      │
│ OVERALL         9.2 █████████▓      │
└─────────────────────────────────────┘

Winner: openai-gpt-4o
Total Cost: $0.0444
```

### Code Review Comparison
```bash
python llmcompare.py --template review --type "Python function" \
  --content "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)" \
  --models '{"openai": ["gpt-4o"], "anthropic": ["claude-3-5-sonnet-latest"]}' --detailed
```

### Business Strategy with Cost Tracking
```bash
python llmcompare.py --template business --topic "AI adoption strategy" \
  --format json --verbose > strategy_analysis.json
```

### Local Models Only
```bash
python llmcompare.py --providers ollama \
  --models '{"ollama": ["llama3.2", "mistral", "codellama"]}' \
  "Explain the difference between REST and GraphQL"
```

### Creative Writing Battle
```bash
python llmcompare.py --template creative --topic "the future of space exploration" \
  --providers openai,anthropic --format markdown > creative_battle.md
```

## 📁 File Structure

```
installation-directory/
├── llmcompare.py                  # Main application
├── .env.template                  # API key template
├── .env                           # Your API keys (create this!)
├── config.yaml                   # Configuration file
├── venv/                         # Python virtual environment
├── logs/                         # Application logs
├── llm-compare.bat               # Windows launcher
├── llm-compare                   # Unix launcher
└── activate.sh/ps1               # Environment activation

~/.llm-compare/
├── config.yaml                  # User configuration
└── history.db                   # Comparison history database
```

## ⚡ Performance & Costs

### Response Times
- **OpenAI**: ~2-5 seconds
- **Anthropic**: ~3-7 seconds  
- **Google**: ~2-4 seconds
- **Ollama**: ~5-30 seconds (depends on model size and hardware)

### Cost Estimates (per 1K tokens)
| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| OpenAI | GPT-4o | $0.005 | $0.015 |
| OpenAI | GPT-4o-mini | $0.00015 | $0.0006 |
| Anthropic | Claude 3.5 Sonnet | $0.003 | $0.015 |
| Google | Gemini 2.0 Flash (Exp) | $0.000075 | $0.0003 |
| Google | Gemini 1.5 Flash | $0.000075 | $0.0003 |
| Ollama | All models | Free | Free |

> **💡 Cost Tip**: Start with GPT-4o-mini or Gemini Flash for experimentation, upgrade to premium models for production use.

## 🔧 Troubleshooting

### Common Issues

**❌ "No providers available"**
```bash
# Check API key configuration
python llmcompare.py --config

# Verify .env file exists and has valid keys
cat .env
```

**Example Error:**
```bash
$ python llmcompare.py "test prompt"

⚠️  No API keys configured!
Run with --config to see setup instructions

$ python llmcompare.py --config

🔧 LLM Comparator Configuration

API Keys Status:
  ✗ OpenAI (GPT models): Not configured
  ✗ Anthropic (Claude models): Not configured
  ✗ Google (Gemini models): Not configured

.env file:
  ✗ Not found. Expected at: /home/user/llm-comparator/.env

Available Providers:
  No providers available
```

**❌ "404 model not found"**
```bash
# Check model names in config
python llmcompare.py --config

# Use current model names:
# ✅ claude-3-5-sonnet-latest (not claude-3-7-sonnet-latest)
# ✅ gemini-1.5-flash (not gemini-pro)
```

**Example Error:**
```bash
$ python llmcompare.py "test prompt"

🔥 LLM Comparison Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model                    ┃ Score ┃ Time   ┃ Cost     ┃ Response Preview               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ google-gemini-pro        │ ERROR │ 2.1s   │ N/A      │ 404 models/gemini-pro not...  │
│ openai-gpt-4o 🏆         │  8.9  │ 2.8s   │ $0.0142  │ This is a test response...     │
└──────────────────────────┴───────┴────────┴──────────┴────────────────────────────────┘

# Fix: Update model name to gemini-1.5-flash
```

**❌ Ollama not working**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Install models
ollama pull llama3.2
```

**Example Error:**
```bash
$ python llmcompare.py --providers ollama "test"

Error: No models available for comparison!

Available providers: []
Requested but unavailable: ['ollama']

Please check your .env file configuration or use --providers to specify available providers.

# Fix: Start Ollama service first
$ ollama serve
# Then try again
$ python llmcompare.py --providers ollama "test"
```

**❌ Ollama not working**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Install models
ollama pull llama3.2
```

**❌ Console buffer errors (during testing)**
```bash
# If you see "Cannot open console output buffer for reading"
# This happens in test environments - run with:
python -c "import os; os.environ['NO_COLOR']='1'; exec(open('test_llmcompare.py').read())"

# Or disable Rich temporarily
export NO_COLOR=1
python test_llmcompare.py
```

**❌ "Model not found" (gemini-2.0-flash)**
```bash
# Check current model names
python llmcompare.py --config

# Use correct model name:
python llmcompare.py --models '{"google": ["gemini-2.0-flash-exp"]}' "prompt"
```

**❌ SQLite database errors**
```bash
# Usually path/permission issues - check:
ls -la ~/.llm-compare/  # Should show history.db

# Or use installation directory
ls -la /path/to/installation/.llm-compare/
```

### Debug Mode

For detailed troubleshooting:
```bash
python llmcompare.py --debug --verbose "test prompt"
```

### Reset Configuration

```bash
# Delete config to reset to defaults
rm ~/.llm-compare/config.yaml

# Run again to regenerate
python llmcompare.py --config
```

## 🎯 Use Cases

### **Developers**
- Compare code explanations across models
- Get multiple perspectives on architectural decisions
- Debug issues with different AI approaches
- Review code quality from various AI viewpoints

### **Researchers**  
- Analyze how different models handle complex topics
- Compare accuracy and depth of explanations
- Study model biases and perspectives
- Generate comprehensive research summaries

### **Writers & Content Creators**
- Get diverse creative approaches to topics
- Compare writing styles and tones
- Generate multiple angles for articles
- Find the most engaging explanations

### **Business Analysts**
- Compare strategic recommendations
- Get multiple perspectives on market analysis
- Evaluate risk assessments from different AI models
- Generate comprehensive business insights

### **Students & Educators**
- Compare explanations for learning concepts
- Get multiple teaching approaches
- Verify information accuracy across models
- Generate study materials from different perspectives

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- **New LLM Providers**: Add support for more APIs
- **Advanced Scoring**: Implement domain-specific scoring
- **UI Improvements**: Enhance the CLI experience
- **Performance**: Optimize concurrent API calls
- **Templates**: Add more specialized prompt templates

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT models and APIs
- **Anthropic** for Claude models  
- **Google** for Gemini models
- **Ollama** for local model infrastructure
- **Rich** library for beautiful CLI interfaces
- The open-source community for inspiration and tools

## 📞 Support

- **Issues**: Open GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check this README and `--help` commands

---

**Happy Comparing! 🚀** 

*Get the best AI responses by testing multiple models simultaneously.*

---

**Made with ❤️ by HumanXAi for the AI community**