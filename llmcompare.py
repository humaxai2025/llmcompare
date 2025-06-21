#!/usr/bin/env python3
"""
LLM Response Comparator - CLI Excellence Version
A feature-rich CLI tool to compare responses from multiple LLMs and score them for quality.
"""

import os
import json
import yaml
import time
import asyncio
import aiohttp
import argparse
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import statistics
import sys
import readline
import atexit
from pathlib import Path

# Rich CLI dependencies
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Install 'rich' for enhanced CLI experience: pip install rich")

# LLM API dependencies
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

def safe_print(message):
    """Safe print that won't fail in test environments"""
    try:
        if RICH_AVAILABLE:
            from rich.console import Console
            console = Console()
            console.print(message)
        else:
            print(message)
    except:
        # Fallback to basic print if Rich fails
        print(message)

@dataclass
class LLMResponse:
    """Data class to store LLM response information"""
    provider: str
    model: str
    response: str
    response_time: float
    token_count: Optional[int] = None
    cost: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Data class to store comparison results"""
    id: str
    prompt: str
    responses: List[LLMResponse]
    scores: Dict[str, Dict[str, float]]
    timestamp: str
    config: Dict[str, Any]
    winner: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConfigManager:
    """Manages configuration files and settings"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".llm-compare"
        self.config_file = self.config_dir / "config.yaml"
        self.config_dir.mkdir(exist_ok=True)
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            "default_models": {
                "openai": ["gpt-4o", "gpt-4o-mini"],
                "anthropic": ["claude-3-7-sonnet-latest"],
                "google": ["gemini-2.0-flash"],
                "ollama": ["llama3.1"]
            },
            "scoring_weights": {
                "relevance": 0.25,
                "clarity": 0.2,
                "completeness": 0.2,
                "accuracy": 0.2,
                "creativity": 0.1,
                "helpfulness": 0.05
            },
            "output": {
                "default_format": "table",
                "color": True,
                "max_response_length": 200
            },
            "api_costs": {
                 "openai": {
                    "gpt-4o": {"input": 0.005, "output": 0.015},
                    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006}
                },
                "anthropic": {
                    "claude-3-5-sonnet-latest": {"input": 0.003, "output": 0.015}
                },
                "google": {
                    "gemini-2.0-flash": {"input": 0.000075, "output": 0.0003},
                   
                }
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                # Merge with defaults
                self.config = {**default_config, **user_config}
            except Exception as error:  # Changed 'e' to 'error'
                pass
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_templates(self):
        """Get prompt templates"""
        return {
            "technical": "Explain {topic} to a software engineer with technical details and code examples where appropriate.",
            "simple": "Explain {topic} in simple terms that a beginner can understand.",
            "creative": "Write a creative and engaging explanation of {topic} using analogies and storytelling.",
            "academic": "Provide a scholarly analysis of {topic} with citations and formal language.",
            "business": "Explain {topic} from a business perspective, focusing on practical applications and ROI.",
            "comparison": "Compare and contrast different aspects of {topic}, highlighting pros and cons.",
            "howto": "Provide step-by-step instructions on how to {action} related to {topic}.",
            "debug": "Help debug this issue: {problem}. Provide systematic troubleshooting steps.",
            "code": "Write {language} code to {task}. Include comments and error handling.",
            "review": "Review this {type}: {content}. Provide constructive feedback and suggestions."
        }


class HistoryManager:
    """Manages command history and comparison results"""
    
    def __init__(self, config_dir: Path):
        self.db_path = config_dir / "history.db"
        self.init_db()
        self.setup_readline()
    
    def init_db(self):
        """Initialize SQLite database for history"""
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS comparisons (
                        id TEXT PRIMARY KEY,
                        prompt TEXT,
                        timestamp TEXT,
                        config TEXT,
                        results TEXT,
                        winner TEXT
                    )
                """)
                # ADD THIS - Missing commands table creation
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS commands (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        command TEXT,
                        timestamp TEXT
                    )
                """)
        except Exception:
            # Use in-memory database as fallback
            self.db_path = ":memory:"
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS comparisons (
                        id TEXT PRIMARY KEY,
                        prompt TEXT,
                        timestamp TEXT,
                        config TEXT,
                        results TEXT,
                        winner TEXT
                    )
                """)
                # ADD THIS TOO - Commands table for in-memory DB
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS commands (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        command TEXT,
                        timestamp TEXT
                    )
                """)
    
    def setup_readline(self):
        """Setup readline for command history"""
        # Skip readline setup for in-memory database
        if self.db_path == ":memory:":
            return

        try:
            history_file = self.db_path.parent / ".llm_compare_history"
            try:
                readline.read_history_file(str(history_file))
            except FileNotFoundError:
                pass

            def save_history():
                readline.write_history_file(str(history_file))

            atexit.register(save_history)
            readline.set_history_length(1000)
        except Exception:
            # Skip readline setup if it fails
            pass
        
    def save_comparison(self, result: ComparisonResult):
        """Save comparison result to history"""
        with sqlite3.connect(self.db_path) as conn:
            # Ensure table exists before inserting
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comparisons (
                    id TEXT PRIMARY KEY,
                    prompt TEXT,
                    timestamp TEXT,
                    config TEXT,
                    results TEXT,
                    winner TEXT
                )
            """)
            
            # Convert to serializable format
            result_dict = {
                'id': result.id,
                'prompt': result.prompt,
                'responses': [asdict(resp) for resp in result.responses],
                'scores': result.scores,
                'timestamp': result.timestamp,
                'config': result.config,
                'winner': result.winner
            }
            
            conn.execute("""
                INSERT OR REPLACE INTO comparisons 
                (id, prompt, timestamp, config, results, winner)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                result.id,
                result.prompt,
                result.timestamp,
                json.dumps(result.config),
                json.dumps(result_dict),
                result.winner
            ))
    def search_history(self, query: str = None, limit: int = 10) -> List[Dict]:
        """Search comparison history"""
        with sqlite3.connect(self.db_path) as conn:
            if query:
                # DEBUG: Print what we're actually searching for
                print(f"DEBUG: Search query received: '{query}' (type: {type(query)})")
                
                # Case-insensitive search
                search_pattern = f"%{query.lower()}%"
                print(f"DEBUG: Search pattern: '{search_pattern}'")
                
                results = conn.execute("""
                    SELECT id, prompt, timestamp, winner 
                    FROM comparisons 
                    WHERE LOWER(prompt) LIKE ? OR LOWER(COALESCE(winner, '')) LIKE ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (search_pattern, search_pattern, limit)).fetchall()
                
                print(f"DEBUG: Found {len(results)} matching results")
            else:
                print("DEBUG: No search query, showing all results")
                results = conn.execute("""
                    SELECT id, prompt, timestamp, winner 
                    FROM comparisons 
                    ORDER BY timestamp DESC LIMIT ?
                """, (limit,)).fetchall()
    
        return [{"id": r[0], "prompt": r[1], "timestamp": r[2], "winner": r[3]} for r in results]
    
    def load_comparison(self, comparison_id: str) -> Optional[ComparisonResult]:
        """Load a specific comparison from history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    SELECT results FROM comparisons WHERE id = ?
                """, (comparison_id,)).fetchone()
            
            if result:
                data = json.loads(result[0])
                # Reconstruct LLMResponse objects
                responses = []
                for resp_data in data.get('responses', []):
                    responses.append(LLMResponse(**resp_data))
                data['responses'] = responses
                return ComparisonResult(**data)
            return None
        except Exception:
            return None


class OutputFormatter:
    """Handles different output formats and styling"""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
    
    def format_table(self, result: ComparisonResult, config: Dict) -> None:
        """Format results as a rich table"""
        table = Table(title="🔥 LLM Comparison Results", show_header=True, header_style="bold magenta")
        
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Score", style="green", justify="center")
        table.add_column("Time", style="yellow", justify="center")
        table.add_column("Cost", style="red", justify="center")
        table.add_column("Response Preview", style="white")
        
        max_length = config.get("output", {}).get("max_response_length", 200)
        
        for response in result.responses:
            if response.error:
                table.add_row(
                    f"{response.provider}-{response.model}",
                    "[red]ERROR[/red]",
                    f"{response.response_time:.2f}s",
                    "N/A",
                    f"[red]{response.error}[/red]"
                )
            else:
                key = f"{response.provider}-{response.model}"
                scores = result.scores.get(key, {})
                avg_score = statistics.mean(scores.values()) if scores else 0
                
                # Determine if this is the winner
                winner_mark = " 🏆" if key == result.winner else ""
                
                # Truncate response for preview
                preview = response.response[:max_length]
                if len(response.response) > max_length:
                    preview += "..."
                
                cost_str = f"${response.cost:.4f}" if response.cost else "N/A"
                
                table.add_row(
                    f"{response.provider}-{response.model}{winner_mark}",
                    f"{avg_score:.2f}",
                    f"{response.response_time:.2f}s",
                    cost_str,
                    preview
                )
        
        self.console.print(table)
    
    def format_detailed_scores(self, result: ComparisonResult) -> None:
        """Display detailed scoring breakdown"""
        for response in result.responses:
            if response.error:
                continue
                
            key = f"{response.provider}-{response.model}"
            scores = result.scores.get(key, {})
            
            if not scores:
                continue
            
            panel_content = []
            for criterion, score in scores.items():
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                panel_content.append(f"{criterion:15} {score:.2f} {bar}")
            
            avg_score = statistics.mean(scores.values())
            panel_content.append(f"{'OVERALL':15} {avg_score:.2f} {'█' * int(avg_score * 10)}")
            
            winner_style = "green" if key == result.winner else "white"
            
            self.console.print(Panel(
                "\n".join(panel_content),
                title=f"{response.provider}-{response.model}",
                border_style=winner_style
            ))
    
    def format_json(self, result: ComparisonResult) -> str:
        """Format results as JSON"""
        return json.dumps(asdict(result), indent=2, default=str)
    
    def format_csv(self, result: ComparisonResult) -> str:
        """Format results as CSV"""
        lines = ["model,provider,score,time,cost,tokens,error"]
        
        for response in result.responses:
            key = f"{response.provider}-{response.model}"
            scores = result.scores.get(key, {})
            avg_score = statistics.mean(scores.values()) if scores else 0
            
            lines.append(f"{response.model},{response.provider},{avg_score:.3f},"
                        f"{response.response_time:.2f},{response.cost or 0},"
                        f"{response.token_count or 0},{response.error or ''}")
        
        return "\n".join(lines)
    
    def format_markdown(self, result: ComparisonResult) -> str:
        """Format results as Markdown"""
        md = [f"# LLM Comparison Results\n"]
        md.append(f"**Prompt:** {result.prompt}\n")
        md.append(f"**Timestamp:** {result.timestamp}\n")
        md.append(f"**Winner:** {result.winner}\n")
        
        md.append("## Results\n")
        md.append("| Model | Score | Time | Cost | Preview |")
        md.append("|-------|-------|------|------|---------|")
        
        for response in result.responses:
            if response.error:
                md.append(f"| {response.provider}-{response.model} | ERROR | {response.response_time:.2f}s | N/A | {response.error} |")
            else:
                key = f"{response.provider}-{response.model}"
                scores = result.scores.get(key, {})
                avg_score = statistics.mean(scores.values()) if scores else 0
                preview = response.response[:100].replace('\n', ' ')
                cost_str = f"${response.cost:.4f}" if response.cost else "N/A"
                
                md.append(f"| {response.provider}-{response.model} | {avg_score:.2f} | {response.response_time:.2f}s | {cost_str} | {preview}... |")
        
        return "\n".join(md)


class EnhancedLLMComparator:
    """Enhanced LLM Comparator with CLI excellence features"""
    # Initialize class attributes to prevent AttributeError
    clients = {}
    debug = False
    verbose = False
    console = None
    config_manager = None

    def __init__(self, args=None):
        # Load .env file if available
        self.load_env_file()
        
        self.config_manager = ConfigManager()
        self.history = HistoryManager(self.config_manager.config_dir)
        self.console = Console() if RICH_AVAILABLE else None
        self.formatter = OutputFormatter(self.console)
        self.args = args
        self.debug = getattr(args, 'debug', False)
        self.verbose = getattr(args, 'verbose', False)

        self.setup_clients()
        
    def check_api_keys_available(self):
        """Check which API keys are available and provide helpful guidance"""
        available_keys = []
        missing_keys = []
        
        # Check for API keys
        api_keys = {
            "OPENAI_API_KEY": "OpenAI (GPT models)",
            "ANTHROPIC_API_KEY": "Anthropic (Claude models)", 
            "GOOGLE_API_KEY": "Google (Gemini models)"
        }
        
        for key, description in api_keys.items():
            value = os.getenv(key)
            provider = key.split('_')[0].lower()
            
            if value and self.is_valid_api_key(value, provider):
                available_keys.append(f"{description}")
                self.debug_print(f"Found valid {key}")
            else:
                missing_keys.append(f"{key} ({description})")
                self.debug_print(f"Missing or invalid {key}")
        
        if not available_keys:
            if self.console and RICH_AVAILABLE:
                self.console.print("\n[yellow]⚠️  No API keys configured![/yellow]")
                self.console.print("\n[blue]To set up API keys:[/blue]")
                self.console.print("1. Copy .env.template to .env in your installation directory")
                self.console.print("2. Edit .env file and replace placeholder keys with your actual keys")
                self.console.print("3. Get API keys from:")
                self.console.print("   • OpenAI: https://platform.openai.com/api-keys")
                self.console.print("   • Anthropic: https://console.anthropic.com/")
                self.console.print("   • Google: https://makersuite.google.com/app/apikey")
                self.console.print("\n[dim]Note: You only need ONE API key to start using the tool![/dim]")
            else:
                print("\n⚠️  No API keys configured!")
                print("\nTo set up API keys:")
                print("1. Copy .env.template to .env in your installation directory")
                print("2. Edit .env file and replace placeholder keys with your actual keys")
                print("3. Get API keys from:")
                print("   • OpenAI: https://platform.openai.com/api-keys")
                print("   • Anthropic: https://console.anthropic.com/")
                print("   • Google: https://makersuite.google.com/app/apikey")
                print("\nNote: You only need ONE API key to start using the tool!")
        else:
            self.verbose_print(f"Available providers: {', '.join(available_keys)}")
            if missing_keys:
                self.debug_print(f"Missing providers: {', '.join(missing_keys)}")
    
    def get_env_file_path(self):
        """Get the path to the .env file for this installation"""
        possible_paths = [
            Path(".env"),
            Path(__file__).parent / ".env",
            Path.home() / "llm-comparator" / ".env",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Return the most likely location for creating a new .env file
        return Path.home() / "llm-comparator" / ".env"
    
    def setup_clients(self):
        self.scoring_criteria = [
            "relevance", "clarity", "completeness", 
            "accuracy", "creativity", "helpfulness"
        ]
        # OpenAI
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai and openai_key and self.is_valid_api_key(openai_key, 'openai'):
            try:
                self.clients['openai'] = openai.OpenAI(api_key=openai_key)
                self.debug_print("OpenAI client initialized successfully")
            except Exception as e:
                self.debug_print(f"OpenAI client initialization failed: {e}")
        else:
            self.debug_print(f"OpenAI client not available (key: {'present' if openai_key else 'missing'})")
            # Don't create client if key is invalid
   
    def load_env_file(self):
        """Load environment variables from .env file if it exists"""
        if not DOTENV_AVAILABLE:
            self.verbose_print("python-dotenv not available, using system environment variables only")
            return
            
        # Look for .env file in current directory and common locations
        possible_env_files = [
            Path(".env"),                                    # Current directory
            Path(__file__).parent / ".env",                 # Script directory
            Path.home() / "llm-comparator" / ".env",        # Installation directory
            Path.cwd() / ".env",                            # Working directory
        ]
        
        env_loaded = False
        for env_file in possible_env_files:
            if env_file and env_file.exists():
                try:
                    load_dotenv(env_file, override=True)
                    self.verbose_print(f"Loaded .env file from {env_file}")
                    env_loaded = True
                    break
                except Exception as e:
                    self.debug_print(f"Failed to load .env from {env_file}: {e}")
        
        if not env_loaded:
            self.verbose_print("No .env file found, using system environment variables")
            
        # Check if we have any API keys now
        self.check_api_keys_available()
    
    def debug_print(self, message: str):
        """Print debug message if debug mode is enabled"""
        if getattr(self, 'debug', False):
            if getattr(self, 'console', None) and RICH_AVAILABLE:
                self.console.print(f"[dim]DEBUG: {message}[/dim]")
            else:
                print(f"DEBUG: {message}")
    
    def verbose_print(self, message: str):
        """Print verbose message if verbose mode is enabled"""
        if getattr(self, 'verbose', False):
            if getattr(self, 'console', None) and RICH_AVAILABLE:
                self.console.print(f"[blue]INFO: {message}[/blue]")
            else:
                print(f"INFO: {message}")
    
    def setup_clients(self):
        """Initialize LLM API clients with better error handling"""
        self.clients = {}
        
        # OpenAI
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai and openai_key and self.is_valid_api_key(openai_key, 'openai'):
            try:
                self.clients['openai'] = openai.OpenAI(api_key=openai_key)
                self.debug_print("OpenAI client initialized successfully")
            except Exception as e:
                self.debug_print(f"OpenAI client initialization failed: {e}")
        else:
            self.debug_print(f"OpenAI client not available (key: {'present' if openai_key else 'missing'})")
        
        # Anthropic
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic and anthropic_key and self.is_valid_api_key(anthropic_key, 'anthropic'):
            try:
                self.clients['anthropic'] = anthropic.Anthropic(api_key=anthropic_key)
                self.debug_print("Anthropic client initialized successfully")
            except Exception as e:
                self.debug_print(f"Anthropic client initialization failed: {e}")
        else:
            self.debug_print(f"Anthropic client not available (key: {'present' if anthropic_key else 'missing'})")
        
        # Google Gemini
       # Google Gemini - REPLACE THIS SECTION:
        google_key = os.getenv('GOOGLE_API_KEY')
        if genai and google_key and self.is_valid_api_key(google_key, 'google'):
            try:
                genai.configure(api_key=google_key)
        # Store the genai module itself, not a specific model
                self.clients['google'] = genai  
                self.debug_print("Google client initialized successfully")
            except Exception as e:
                self.debug_print(f"Google client initialization failed: {e}")
        else:
                self.debug_print(f"Google client not available (key: {'present' if google_key else 'missing'})")


        try:
        # Test if Ollama is running by making a simple request
            import aiohttp
            import asyncio
        
            async def test_ollama():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get('http://localhost:11434/api/tags', timeout=2) as response:
                            if response.status == 200:
                                return True
                    return False
                except:
                    return False
        
            # Run the async test in a sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ollama_available = loop.run_until_complete(test_ollama())
            loop.close()
        
            if ollama_available:
                self.clients['ollama'] = True  # Just mark as available
                self.debug_print("Ollama client initialized successfully")
            else:
                self.debug_print("Ollama not available - service not running on localhost:11434")
            
        except Exception as e:
            self.debug_print(f"Ollama client initialization failed: {e}")








      
        
        # Check if we have any working clients
        if not self.clients:
            self.verbose_print("No API clients available - please check your .env file configuration")
        else:
            self.verbose_print(f"Initialized {len(self.clients)} API client(s): {', '.join(self.clients.keys())}")
    
    def is_valid_api_key(self, key: str, provider: str) -> bool:
        """Check if an API key appears to be valid (not a placeholder)"""
        if not key or not isinstance(key, str):
            return False
        
        # Check for common placeholder patterns
        if any(placeholder in key.lower() for placeholder in ['your-', 'insert-', 'add-your', 'replace-', 'enter-']):
            return False
        
        # Check for minimum length
        if len(key.strip()) < 10:
            return False
        
        # Provider-specific validation
        if provider == 'openai':
            # OpenAI keys typically start with 'sk-'
            return key.startswith('sk-') and len(key) > 20
        elif provider == 'anthropic':
            # Anthropic keys can start with 'sk-ant-' or other patterns
            return (key.startswith('sk-ant-') or key.startswith('sk-')) and len(key) > 20
        elif provider == 'google':
            # Google API keys have various formats, often start with 'AIza' but not always
            return len(key) > 15 and not key.startswith('your-')
        
        return True
    
    def calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost based on token usage"""
        costs = self.config_manager.config.get("api_costs", {})
        provider_costs = costs.get(provider, {})
        model_costs = provider_costs.get(model, {})
        
        if "input" in model_costs and "output" in model_costs:
            return (input_tokens * model_costs["input"] / 1000) + (output_tokens * model_costs["output"] / 1000)
        return 0.0
    
    async def query_openai(self, prompt: str, model: str = "gpt-4o") -> LLMResponse:
        """Query OpenAI models"""
        if 'openai' not in self.clients:
            return LLMResponse("openai", model, "", 0, error="OpenAI client not available")
        
        try:
            self.verbose_print(f"Querying OpenAI {model}")
            start_time = time.time()
            response = self.clients['openai'].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            end_time = time.time()
            
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost = self.calculate_cost("openai", model, input_tokens, output_tokens)
            
            return LLMResponse(
                provider="openai",
                model=model,
                response=response.choices[0].message.content,
                response_time=end_time - start_time,
                token_count=response.usage.total_tokens if response.usage else None,
                cost=cost
            )
        except Exception as e:
            self.debug_print(f"OpenAI error: {e}")
            return LLMResponse("openai", model, "", 0, error=str(e))
    
    async def query_anthropic(self, prompt: str, model: str = "claude-3-7-sonnet-latest") -> LLMResponse:
        """Query Anthropic Claude models"""
        if 'anthropic' not in self.clients:
            return LLMResponse("anthropic", model, "", 0, error="Anthropic client not available")
        
        try:
            self.verbose_print(f"Querying Anthropic {model}")
            start_time = time.time()
            response = self.clients['anthropic'].messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            end_time = time.time()
            
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0
            cost = self.calculate_cost("anthropic", model, input_tokens, output_tokens)
            
            return LLMResponse(
                provider="anthropic",
                model=model,
                response=response.content[0].text,
                response_time=end_time - start_time,
                token_count=input_tokens + output_tokens,
                cost=cost
            )
        except Exception as e:
            self.debug_print(f"Anthropic error: {e}")
            return LLMResponse("anthropic", model, "", 0, error=str(e))
    
    async def query_google(self, prompt: str, model: str = "gemini-2.0-flash") -> LLMResponse:
        """Query Google Gemini models"""
        if 'google' not in self.clients:
            return LLMResponse("google", model, "", 0, error="Google client not available")
    
        try:
            self.verbose_print(f"Querying Google {model}")
            start_time = time.time()
        
        # Create model instance for this specific request
            google_model = self.clients['google'].GenerativeModel(model)
            response = google_model.generate_content(prompt)
        
            end_time = time.time()

            # Extract token usage if available
            input_tokens = 0
            output_tokens = 0

            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                self.debug_print(f"Google tokens: input={input_tokens}, output={output_tokens}")

            cost = self.calculate_cost("google", model, input_tokens, output_tokens)
            return LLMResponse(
                provider="google",
                model=model,
                response=response.text,
                response_time=end_time - start_time,
                token_count=None,
                cost=cost  # Google pricing varies
            )
        except Exception as e:
            self.debug_print(f"Google error: {e}")
            self.debug_print(f"Could not extract Google token usage: {e}")
            return LLMResponse("google", model, "", 0, error=str(e))
    
    async def query_ollama(self, prompt: str, model: str = "llama3.1") -> LLMResponse:
        """Query local Ollama models"""
        try:
            self.verbose_print(f"Querying Ollama {model}")
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'http://localhost:11434/api/generate',
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        end_time = time.time()
                        return LLMResponse(
                            provider="ollama",
                            model=model,
                            response=data.get('response', ''),
                            response_time=end_time - start_time,
                            cost=0.0  # Local models are free
                        )
                    else:
                        return LLMResponse("ollama", model, "", 0, error=f"HTTP {response.status}")
        except Exception as e:
            self.debug_print(f"Ollama error: {e}")
            return LLMResponse("ollama", model, "", 0, error=str(e))
    
    def score_response_basic(self, response: str, prompt: str) -> Dict[str, float]:
        """Basic scoring based on response characteristics"""
        scores = {}
        
        # Length appropriateness (not too short, not too verbose)
        length = len(response.split())
        if 50 <= length <= 500:
            scores['length_appropriateness'] = 1.0
        elif 20 <= length < 50 or 500 < length <= 1000:
            scores['length_appropriateness'] = 0.7
        else:
            scores['length_appropriateness'] = 0.3
        
        # Structure score (presence of paragraphs, sentences)
        sentences = response.count('.') + response.count('!') + response.count('?')
        paragraphs = response.count('\n\n') + 1
        scores['structure'] = min(1.0, (sentences / max(1, paragraphs)) * 0.2)
        
        # Relevance to prompt (simple keyword matching)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words.intersection(response_words))
        scores['keyword_relevance'] = min(1.0, overlap / max(1, len(prompt_words)))
        
        # Completeness (does it seem to address the prompt)
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        prompt_questions = sum(1 for word in question_words if word in prompt.lower())
        if prompt_questions > 0:
            scores['completeness'] = 0.8 if len(response) > 50 else 0.3
        else:
            scores['completeness'] = 0.7
        
        return scores
    
    async def score_with_llm_judge(self, responses: List[LLMResponse], prompt: str) -> Dict[str, Dict[str, float]]:
        """Use an LLM as a judge to score responses"""
        if 'openai' not in self.clients:
            return {}
        
        self.verbose_print("Running LLM judge evaluation")
        
        judge_prompt = f"""
        You are an expert evaluator of AI responses. Rate the following responses to the prompt on a scale of 1-10 for each criterion.
        
        Original Prompt: "{prompt}"
        
        Criteria to evaluate:
        1. Relevance - How well does it address the prompt?
        2. Clarity - How clear and understandable is the response?
        3. Completeness - How thoroughly does it answer the question?
        4. Accuracy - How factually correct does it appear?
        5. Creativity - How creative or insightful is the response?
        6. Helpfulness - How useful would this be to the user?
        
        Responses to evaluate:
        """
        
        for i, resp in enumerate(responses):
            if not resp.error:
                judge_prompt += f"\nResponse {i+1} ({resp.provider}-{resp.model}):\n{resp.response}\n"
        
        judge_prompt += "\nProvide scores in JSON format like: {\"response_1\": {\"relevance\": 8, \"clarity\": 7, ...}, \"response_2\": {...}}"
        
        try:
            response = self.clients['openai'].chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=1000
            )
            
            # Parse JSON response
            scores_text = response.choices[0].message.content
            scores_start = scores_text.find('{')
            scores_end = scores_text.rfind('}') + 1
            scores_json = scores_text[scores_start:scores_end]
            
            scores = json.loads(scores_json)
            
            # Convert to provider-model keys
            result = {}
            for i, resp in enumerate(responses):
                if not resp.error:
                    key = f"{resp.provider}-{resp.model}"
                    response_key = f"response_{i+1}"
                    if response_key in scores:
                        result[key] = {k: v/10.0 for k, v in scores[response_key].items()}
            
            return result
            
        except Exception as e:
            self.debug_print(f"LLM judge scoring failed: {e}")
            return {}
    
    async def compare_responses(self, prompt: str, models: Optional[Dict[str, List[str]]] = None) -> ComparisonResult:
        """Compare responses from multiple LLMs"""
        if models is None:
            models = self.config_manager.config["default_models"]
        
        # Check if we have any available clients
        if not self.clients:
            error_msg = """
No API clients available! Please set up your API keys.

Quick setup:
1. Copy .env.template to .env in your installation directory
2. Edit .env and add your API keys
3. Get keys from:
   • OpenAI: https://platform.openai.com/api-keys
   • Anthropic: https://console.anthropic.com/
   • Google: https://makersuite.google.com/app/apikey

Note: You only need ONE API key to start using the tool!
"""
            raise Exception(error_msg.strip())
        
        # Generate comparison ID
        comparison_id = f"comp_{int(time.time())}"
        
        # Generate all tasks
        tasks = []
        skipped_providers = []
        
        for provider, model_list in models.items():
            if provider not in self.clients:
                skipped_providers.append(provider)
                self.verbose_print(f"Skipping {provider} - not available")
                continue
                
            for model in model_list:
                if provider == 'openai':
                    tasks.append(self.query_openai(prompt, model))
                elif provider == 'anthropic':
                    tasks.append(self.query_anthropic(prompt, model))
                elif provider == 'google':
                    tasks.append(self.query_google(prompt, model))
                elif provider == 'ollama':
                    tasks.append(self.query_ollama(prompt, model))
        
        if not tasks:
            available = list(self.clients.keys())
            skipped = list(set(models.keys()) - set(available))
            error_msg = f"""
No models available for comparison!

Available providers: {available}
Requested but unavailable: {skipped}

Please check your .env file configuration or use --providers to specify available providers.
Example: llm-compare --providers {','.join(available)} "your prompt"
"""
            raise Exception(error_msg.strip())
        
        self.verbose_print(f"Querying {len(tasks)} models...")
        if skipped_providers:
            self.verbose_print(f"Skipped providers: {', '.join(skipped_providers)}")
        
        # Run with progress bar if rich is available
        if RICH_AVAILABLE and self.console and not getattr(self.args, 'no_progress', False):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Querying models...", total=len(tasks))
                
                responses = []
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    response = await coro
                    responses.append(response)
                    progress.update(task, advance=1)
        else:
            responses = await asyncio.gather(*tasks)
        
        # Filter out failed responses for scoring
        successful_responses = [r for r in responses if not r.error]
        
        if not successful_responses:
            raise Exception("All LLM queries failed. Please check your API keys and network connection.")
        
        self.verbose_print("Scoring responses...")
        
        # Combine basic scores with LLM judge scores
        all_scores = {}
        
        # Basic scoring for all responses
        for response in successful_responses:
            key = f"{response.provider}-{response.model}"
            basic_scores = self.score_response_basic(response.response, prompt)
            all_scores[key] = basic_scores
        
        # LLM judge scoring
        llm_scores = await self.score_with_llm_judge(successful_responses, prompt)
        
        # Merge scores
        for key in llm_scores:
            if key in all_scores:
                all_scores[key].update(llm_scores[key])
        
        # Apply scoring weights
        weights = self.config_manager.config["scoring_weights"]
        weighted_scores = {}
        for key, scores in all_scores.items():
            weighted_score = 0
            total_weight = 0
            for criterion, score in scores.items():
                weight = weights.get(criterion, 1.0)
                weighted_score += score * weight
                total_weight += weight
            weighted_scores[key] = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine winner
        winner = max(weighted_scores.keys(), key=lambda k: weighted_scores[k]) if weighted_scores else None
        
        result = ComparisonResult(
            id=comparison_id,
            prompt=prompt,
            responses=responses,
            scores=all_scores,
            timestamp=datetime.now().isoformat(),
            config=self.config_manager.config,
            winner=winner
        )
        
        # Save to history
        self.history.save_comparison(result)
        
        return result
    
    def apply_template(self, template_name: str, **kwargs) -> str:
        """Apply a prompt template with given parameters"""
        templates = self.config_manager.get_templates()
        if template_name not in templates:
            available = ", ".join(templates.keys())
            raise ValueError(f"Template '{template_name}' not found. Available: {available}")
        
        template = templates[template_name]
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template requires parameter: {e}")
    
    def run_interactive_mode(self):
        """Run interactive CLI mode"""
        if self.console:
            self.console.print(Panel.fit(
                "🚀 LLM Comparator Interactive Mode\n"
                "Commands: compare, history, config, templates, help, quit",
                title="Welcome",
                border_style="blue"
            ))
        else:
            print("=== LLM Comparator Interactive Mode ===")
            print("Commands: compare, history, config, templates, help, quit")
        
        while True:
            try:
                if RICH_AVAILABLE:
                    command = Prompt.ask("\n[bold cyan]llm-compare[/bold cyan]", default="compare")
                else:
                    command = input("\nllm-compare> ")
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.lower() in ['help', 'h']:
                    self.show_help()
                elif command.lower().startswith('compare'):
                    self.handle_interactive_compare(command)
                elif command.lower().startswith('history'):
                    self.handle_interactive_history(command)
                elif command.lower() in ['config']:
                    self.show_config()
                elif command.lower() in ['templates']:
                    self.show_templates()
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def handle_interactive_compare(self, command: str):
        """Handle compare command in interactive mode"""
        if RICH_AVAILABLE:
            prompt = Prompt.ask("Enter your prompt")
            use_template = Confirm.ask("Use a template?", default=False)
            
            if use_template:
                templates = list(self.config_manager.get_templates().keys())
                self.console.print("Available templates:", ", ".join(templates))
                template_name = Prompt.ask("Template name")
                
                if template_name in self.config_manager.get_templates():
                    # Get template parameters
                    template = self.config_manager.get_templates()[template_name]
                    import re
                    params = re.findall(r'\{(\w+)\}', template)
                    
                    kwargs = {}
                    for param in params:
                        kwargs[param] = Prompt.ask(f"Enter {param}")
                    
                    prompt = self.apply_template(template_name, **kwargs)
                    self.console.print(f"Generated prompt: {prompt}")
        else:
            prompt = input("Enter your prompt: ")
        
        # Run comparison
        result = asyncio.run(self.compare_responses(prompt))
        self.display_result(result)
    
    def handle_interactive_history(self, command: str):
        """Handle history command in interactive mode"""
        parts = command.split()
        query = parts[1] if len(parts) > 1 else None
        
        history = self.history.search_history(query)
        
        if self.console:
            table = Table(title="Recent Comparisons")
            table.add_column("ID", style="cyan")
            table.add_column("Prompt", style="white")
            table.add_column("Winner", style="green")
            table.add_column("Time", style="yellow")
            
            for item in history:
                table.add_row(
                    item["id"][-8:],  # Show last 8 chars of ID
                    item["prompt"][:50] + "..." if len(item["prompt"]) > 50 else item["prompt"],
                    item["winner"] or "N/A",
                    item["timestamp"][:19]  # Remove milliseconds
                )
            
            self.console.print(table)
        else:
            print("Recent Comparisons:")
            for item in history:
                print(f"  {item['id'][-8:]}: {item['prompt'][:50]}... -> {item['winner']}")
    
    def show_help(self):
        """Show help information"""
        help_text = """
Available Commands:
  compare          - Run a new comparison
  history [query]  - Show comparison history
  config          - Show current configuration
  templates       - Show available prompt templates
  help            - Show this help
  quit            - Exit interactive mode

Examples:
  compare
  history AI
  templates
        """
        
        if self.console:
            self.console.print(Panel(help_text, title="Help", border_style="green"))
        else:
            print(help_text)
    
    def show_config(self):
        """Show current configuration including API key status"""
        env_file_path = self.get_env_file_path()
        
        if self.console:
            # Rich formatted output
            self.console.print(Panel.fit(
                "🔧 LLM Comparator Configuration", 
                title="Configuration", 
                border_style="blue"
            ))
            
            # API Keys Status
            self.console.print("\n[bold cyan]API Keys Status:[/bold cyan]")
            api_keys = {
                "OPENAI_API_KEY": "OpenAI (GPT models)",
                "ANTHROPIC_API_KEY": "Anthropic (Claude models)", 
                "GOOGLE_API_KEY": "Google (Gemini models)"
            }
            
            for key, description in api_keys.items():
                value = os.getenv(key)
                provider = key.split('_')[0].lower()
                
                if value and self.is_valid_api_key(value, provider):
                    self.console.print(f"  ✓ {description}: [green]Configured[/green]")
                else:
                    self.console.print(f"  ✗ {description}: [red]Not configured[/red]")
            
            # .env file location
            self.console.print(f"\n[bold cyan].env file:[/bold cyan]")
            if env_file_path.exists():
                self.console.print(f"  ✓ Found at: [green]{env_file_path}[/green]")
            else:
                self.console.print(f"  ✗ Not found. Expected at: [yellow]{env_file_path}[/yellow]")
            
            # Available providers
            self.console.print(f"\n[bold cyan]Available Providers:[/bold cyan]")
            if self.clients:
                for provider in self.clients.keys():
                    self.console.print(f"  ✓ [green]{provider}[/green]")
            else:
                self.console.print("  [red]No providers available[/red]")
            
            # Configuration file
            config_yaml = yaml.dump(self.config_manager.config, default_flow_style=False)
            syntax = Syntax(config_yaml, "yaml", theme="monokai", line_numbers=True)
            self.console.print(Panel(syntax, title="config.yaml", border_style="blue"))
            
        else:
            # Plain text output
            print("=== LLM Comparator Configuration ===")
            print()
            print("API Keys Status:")
            api_keys = {
                "OPENAI_API_KEY": "OpenAI (GPT models)",
                "ANTHROPIC_API_KEY": "Anthropic (Claude models)", 
                "GOOGLE_API_KEY": "Google (Gemini models)"
            }
            
            for key, description in api_keys.items():
                value = os.getenv(key)
                provider = key.split('_')[0].lower()
                
                if value and self.is_valid_api_key(value, provider):
                    print(f"  ✓ {description}: Configured")
                else:
                    print(f"  ✗ {description}: Not configured")
            
            print()
            print(f".env file:")
            if env_file_path.exists():
                print(f"  ✓ Found at: {env_file_path}")
            else:
                print(f"  ✗ Not found. Expected at: {env_file_path}")
            
            print()
            print("Available Providers:")
            if self.clients:
                for provider in self.clients.keys():
                    print(f"  ✓ {provider}")
            else:
                print("  No providers available")
            
            print()
            print("Configuration:")
            print(yaml.dump(self.config_manager.config, default_flow_style=False))
    
    def show_setup_help(self):
        """Show API key setup instructions"""
        env_file_path = self.get_env_file_path()
    
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel.fit(
                "🔑 API Key Setup Instructions", 
                title="Setup Help", 
                border_style="green"
            ))
            
            self.console.print("\n[bold green]Step 1: Create .env file[/bold green]")
            if (env_file_path.parent / ".env.template").exists():
                self.console.print(f"Copy the template: [cyan]copy \"{env_file_path.parent / '.env.template'}\" \"{env_file_path}\"[/cyan]")
            else:
                self.console.print(f"Create a new file: [cyan]{env_file_path}[/cyan]")
            
            self.console.print("\n[bold green]Step 2: Add your API keys to .env file[/bold green]")
            self.console.print("Edit the .env file and add:")
            self.console.print("[cyan]OPENAI_API_KEY=sk-your-actual-openai-key[/cyan]")
            self.console.print("[cyan]ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key[/cyan]")
            self.console.print("[cyan]GOOGLE_API_KEY=your-actual-google-key[/cyan]")
            
            self.console.print("\n[bold green]Step 3: Get API keys[/bold green]")
            self.console.print("• OpenAI: [link]https://platform.openai.com/api-keys[/link]")
            self.console.print("• Anthropic: [link]https://console.anthropic.com/[/link]")
            self.console.print("• Google: [link]https://makersuite.google.com/app/apikey[/link]")
            
            self.console.print("\n[bold yellow]Note:[/bold yellow] You only need ONE API key to start using the tool!")
            
        else:
            print("=== API Key Setup Instructions ===")
            print()
            print("Step 1: Create .env file")
            if (env_file_path.parent / ".env.template").exists():
                print(f"Copy the template: copy \"{env_file_path.parent / '.env.template'}\" \"{env_file_path}\"")
            else:
                print(f"Create a new file: {env_file_path}")
            
            print()
            print("Step 2: Add your API keys to .env file")
            print("Edit the .env file and add:")
            print("OPENAI_API_KEY=sk-your-actual-openai-key")
            print("ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key")  
            print("GOOGLE_API_KEY=your-actual-google-key")
            
            print()
            print("Step 3: Get API keys")
            print("• OpenAI: https://platform.openai.com/api-keys")
            print("• Anthropic: https://console.anthropic.com/")
            print("• Google: https://makersuite.google.com/app/apikey")
            
            print()
            print("Note: You only need ONE API key to start using the tool!")
    
    def show_templates(self):
        """Show available templates"""
        templates = self.config_manager.get_templates()
        
        if self.console:
            for name, template in templates.items():
                self.console.print(f"[bold cyan]{name}[/bold cyan]: {template}")
        else:
            print("Available Templates:")
            for name, template in templates.items():
                print(f"  {name}: {template}")
    
    def display_result(self, result: ComparisonResult):
        """Display comparison result based on output format"""
        output_format = getattr(self.args, 'format', 'table')
        
        if output_format == 'table':
            self.formatter.format_table(result, self.config_manager.config)
            if getattr(self.args, 'detailed', False):
                self.formatter.format_detailed_scores(result)
        elif output_format == 'json':
            print(self.formatter.format_json(result))
        elif output_format == 'csv':
            print(self.formatter.format_csv(result))
        elif output_format == 'markdown':
            if self.console:
                md = Markdown(self.formatter.format_markdown(result))
                self.console.print(md)
            else:
                print(self.formatter.format_markdown(result))


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Compare responses from multiple LLMs with rich CLI features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llm-compare "Explain quantum computing"
  llm-compare --template technical --topic "machine learning" 
  llm-compare --interactive
  llm-compare --format json "What is AI?" > results.json
  llm-compare --history --search "quantum"
  llm-compare --config
  llm-compare --setup-help
        """
    )
    
    # Main arguments
    parser.add_argument('prompt', nargs='?', help='Prompt to compare (or use --interactive)')
    
    # Templates
    parser.add_argument('--template', choices=[
        'technical', 'simple', 'creative', 'academic', 'business', 
        'comparison', 'howto', 'debug', 'code', 'review'
    ], help='Use a prompt template')
    parser.add_argument('--topic', help='Topic for template (required with --template)')
    parser.add_argument('--action', help='Action for howto template')
    parser.add_argument('--problem', help='Problem for debug template')
    parser.add_argument('--language', help='Language for code template')
    parser.add_argument('--task', help='Task for code template')
    parser.add_argument('--type', help='Type for review template')
    parser.add_argument('--content', help='Content for review template')
    
    # Output formatting
    parser.add_argument('--format', choices=['table', 'json', 'csv', 'markdown'], 
                       default='table', help='Output format')
    parser.add_argument('--detailed', action='store_true', 
                       help='Show detailed scoring breakdown')
    parser.add_argument('--no-color', action='store_true', 
                       help='Disable colored output')
    
    # Models
    parser.add_argument('--models', help='JSON string of models to use')
    parser.add_argument('--providers', help='Comma-separated list of providers to use')
    
    # Modes
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--config', action='store_true', 
                       help='Show configuration and API key status')
    parser.add_argument('--templates', action='store_true', 
                       help='Show available templates and exit')
    parser.add_argument('--setup-help', action='store_true',
                       help='Show API key setup instructions')
    
    # History
    parser.add_argument('--history', action='store_true', 
                       help='Show comparison history')
    parser.add_argument('--search', help='Search history for term')
    parser.add_argument('--load', help='Load comparison by ID')
    
    # Debug
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bars')
    
    return parser


def main():
    """Main CLI entry point"""
    try:
        parser = create_parser()
        args = parser.parse_args()
        
        print("Creating comparator...")  # Debug
        comparator = EnhancedLLMComparator(args)
        
        print("Setting up clients...")  # Debug
        comparator.setup_clients()
        
        print("Processing arguments...")  # Debug
        # ... rest of your code
        
    except Exception as error:
        print(f"Error occurred: {error}")
        import traceback
        traceback.print_exc()
    
    # Initialize comparator
    comparator = EnhancedLLMComparator(args)

    comparator.setup_clients()
    
    try:
        # Handle special modes
        if args.config:
            comparator.show_config()
            return
        
        if args.templates:
            comparator.show_templates()
            return
        
        if args.setup_help:
            comparator.show_setup_help()
            return
        
        # Check if user needs help with API keys
        if not comparator.clients:
            if not (args.history or args.load):
                if comparator.console and RICH_AVAILABLE:
                    comparator.console.print("\n[red]No API keys configured![/red]")
                    comparator.console.print("[yellow]Run with --config to see setup instructions[/yellow]")
                else:
                    print("\nNo API keys configured!")
                    print("Run with --config to see setup instructions")
                return
        
        if args.history:
             # FIX: Ensure search parameter is properly passed
            search_query = getattr(args, 'search', None)
            history = comparator.history.search_history(search_query)
    
            
            if comparator.console:
                    table = Table(title="Comparison History")
                    table.add_column("ID", style="cyan")
                    table.add_column("Prompt", style="white") 
                    table.add_column("Winner", style="green")
                    table.add_column("Time", style="yellow")
                    
                    for item in history:
                        table.add_row(
                            item["id"][-8:],
                            item["prompt"][:60] + "..." if len(item["prompt"]) > 60 else item["prompt"],
                            item["winner"] or "N/A",
                            item["timestamp"][:19]
                        )
                    
                    comparator.console.print(table)
            else:
                for item in history:
                 print(f"{item['id'][-8:]}: {item['prompt'][:60]}... -> {item['winner']}")
            return
        
        if args.load:
            result = comparator.history.load_comparison(args.load)
            if result:
                comparator.display_result(result)
            else:
                print(f"Comparison {args.load} not found")
            return
        
        if args.interactive:
            comparator.run_interactive_mode()
            return
        
        # Handle template mode
        if args.template:
            required_params = {
                'technical': ['topic'],
                'simple': ['topic'],
                'creative': ['topic'],
                'academic': ['topic'],
                'business': ['topic'],
                'comparison': ['topic'],
                'howto': ['action', 'topic'],
                'debug': ['problem'],
                'code': ['language', 'task'],
                'review': ['type', 'content']
            }
            
            params = {}
            for param in required_params.get(args.template, []):
                value = getattr(args, param, None)
                if not value:
                    print(f"Error: --{param} is required for template '{args.template}'")
                    return
                params[param] = value
            
            prompt = comparator.apply_template(args.template, **params)
            if args.verbose:
                print(f"Generated prompt: {prompt}")
        elif args.prompt:
            prompt = args.prompt
        else:
            print("Error: Provide a prompt or use --interactive mode")
            parser.print_help()
            return
        
        # Parse models if provided
        models = None
        if args.models:
            try:
                models = json.loads(args.models)
            except json.JSONDecodeError:
                print("Error: Invalid JSON for --models")
                return
        
        if args.providers:
            provider_list = [p.strip() for p in args.providers.split(',')]
            default_models = comparator.config_manager.config["default_models"]
            models = {p: default_models.get(p, []) for p in provider_list if p in default_models}
        
        # Run comparison
        result = asyncio.run(comparator.compare_responses(prompt, models))
        comparator.display_result(result)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()