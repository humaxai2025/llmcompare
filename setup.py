#!/usr/bin/env python3
"""
LLM Comparator Simple Installer - Updated Version
Cross-platform installer that works on Windows, Mac, and Linux
Stores configuration locally in installation directory
No PowerShell, no admin rights needed
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import json
from pathlib import Path

def print_status(message):
    print(f"[INFO] {message}")

def print_success(message):
    print(f"[SUCCESS] {message}")

def print_warning(message):
    print(f"[WARNING] {message}")

def print_error(message):
    print(f"[ERROR] {message}")

def check_python():
    """Check if Python version is adequate"""
    if sys.version_info < (3, 8):
        print_error(f"Python 3.8+ required. Found: {sys.version}")
        print("Please install Python from: https://www.python.org/downloads/")
        return False
    
    print_success(f"Python {sys.version.split()[0]} found")
    return True

def get_install_paths():
    """Get installation paths based on OS - UPDATED to store config locally"""
    home = Path.home()
    
    if platform.system() == "Windows":
        install_dir = home / "llm-comparator"
    else:
        install_dir = home / "llm-comparator"
    
    # CHANGED: Store config in installation directory, not user home
    config_dir = install_dir / ".llm-compare"
    
    return install_dir, config_dir

def create_directories(install_dir, config_dir):
    """Create necessary directories"""
    print_status("Creating directories...")
    
    install_dir.mkdir(exist_ok=True)
    config_dir.mkdir(exist_ok=True)
    (install_dir / "logs").mkdir(exist_ok=True)
    (install_dir / "backups").mkdir(exist_ok=True)
    
    print_success("Directories created")

def setup_virtual_environment(install_dir):
    """Create Python virtual environment"""
    print_status("Setting up Python virtual environment...")
    
    venv_dir = install_dir / "venv"
    
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        print_success("Virtual environment created")
        return venv_dir
    except subprocess.CalledProcessError:
        print_error("Failed to create virtual environment")
        return None

def get_python_executable(venv_dir):
    """Get the Python executable path for the virtual environment"""
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    else:
        return venv_dir / "bin" / "python"

def install_dependencies(python_exe):
    """Install required Python packages"""
    print_status("Installing dependencies...")
    
    packages = [
        "aiohttp>=3.8.0",
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "google-generativeai>=0.3.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0"
    ]
    
    try:
        for package in packages:
            print(f"  Installing {package}...")
            subprocess.run([str(python_exe), "-m", "pip", "install", package], 
                         check=True, capture_output=True)
        
        print_success("All dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def create_config_file(config_dir):
    """Create default configuration file"""
    print_status("Creating configuration file...")
    
    config = {
        "default_models": {
            "openai": ["gpt-4o", "gpt-4o-mini"],
            "anthropic": ["claude-3-5-sonnet-latest"],
            "google": ["gemini-1.5-flash"],
            "ollama": ["llama3.2"]
        },
        "scoring_weights": {
            "relevance": 0.25,
            "clarity": 0.20,
            "completeness": 0.20,
            "accuracy": 0.20,
            "creativity": 0.10,
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
                "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
                "gemini-1.5-pro": {"input": 0.00125, "output": 0.005}
            }
        },
        "custom_templates": {
            "explain_code": "Explain this {language} code step by step: {code}",
            "debug_issue": "Help me debug this {language} issue: {problem}",
            "optimize_code": "Suggest optimizations for this {language} code: {code}",
            "code_review": "Review this {language} code for best practices: {code}"
        }
    }
    
    try:
        import yaml
        with open(config_dir / "config.yaml", "w", encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        print_success("Configuration file created")
        return True
    except Exception as e:
        print_error(f"Failed to create config file: {e}")
        return False

def create_launcher_scripts(install_dir, venv_dir):
    """Create launcher scripts for different platforms"""
    print_status("Creating launcher scripts...")
    
    python_exe = get_python_executable(venv_dir)
    
    if platform.system() == "Windows":
        # Windows batch file - FIXED
        batch_content = f"""@echo off
cd /d "{install_dir}"
"{python_exe}" "{install_dir}\\llmcompare.py" %*
pause
"""
        
        with open(install_dir / "llm-compare.bat", "w", encoding='utf-8') as f:
            f.write(batch_content)
        
        # PowerShell script - FIXED
        ps_content = f"""function llm-compare {{
    param([Parameter(ValueFromRemainingArguments)]$Args)
    Set-Location "{install_dir}"
    & "{python_exe}" "{install_dir}\\llmcompare.py" @Args
}}

Write-Host "LLM Comparator activated. Try: llm-compare --help" -ForegroundColor Green
"""
        
        with open(install_dir / "activate.ps1", "w", encoding='utf-8') as f:
            f.write(ps_content)
        
        # Simple direct launcher - NEW
        direct_launcher = f"""@echo off
"{python_exe}" "{install_dir}\\llmcompare.py" %*
"""
        
        with open(install_dir / "run-llm-compare.bat", "w", encoding='utf-8') as f:
            f.write(direct_launcher)
            
    else:
        # Unix shell script - FIXED  
        shell_content = f"""#!/bin/bash
cd "{install_dir}"
"{python_exe}" "{install_dir}/llmcompare.py" "$@"
"""
        
        launcher = install_dir / "llm-compare"
        with open(launcher, "w", encoding='utf-8') as f:
            f.write(shell_content)
        launcher.chmod(0o755)
        
        # Direct launcher - NEW
        direct_launcher = install_dir / "run-llm-compare.sh"
        direct_content = f"""#!/bin/bash
"{python_exe}" "{install_dir}/llmcompare.py" "$@"
"""
        
        with open(direct_launcher, "w", encoding='utf-8') as f:
            f.write(direct_content)
        direct_launcher.chmod(0o755)
        
        # Activation script - FIXED
        activate_content = f"""#!/bin/bash
export LLM_COMPARATOR_HOME="{install_dir}"
export LLM_COMPARATOR_VENV="{venv_dir}"

llm-compare() {{
    cd "{install_dir}"
    "{python_exe}" "{install_dir}/llmcompare.py" "$@"
}}

echo "LLM Comparator activated. Try: llm-compare --help"
"""
        
        with open(install_dir / "activate.sh", "w", encoding='utf-8') as f:
            f.write(activate_content)
    
    print_success("Launcher scripts created")

def create_env_file(install_dir):
    """Create .env template file for API keys"""
    print_status("Creating .env template file...")
    
    env_content = """# LLM Comparator API Keys
# Copy this file to .env and add your actual API keys

# OpenAI API Key (get from: https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-your-openai-key-here

# Anthropic API Key (get from: https://console.anthropic.com/)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Google API Key (get from: https://makersuite.google.com/app/apikey)
GOOGLE_API_KEY=your-google-key-here

# Optional: Ollama settings (for local models)
OLLAMA_HOST=http://localhost:11434

# Usage:
# 1. Remove this comment section
# 2. Replace the placeholder keys with your actual API keys
# 3. Save this file as .env (remove .template)
# 4. Keep this file secure and never share it publicly!
"""

    try:
        with open(install_dir / ".env.template", "w", encoding='utf-8') as f:
            f.write(env_content)
        
        print_success(".env template created")
        print_warning("Remember to copy .env.template to .env and add your API keys!")
        return True
    except Exception as e:
        print_error(f"Failed to create .env template: {e}")
        return False

def check_api_keys(install_dir):
    """Check for API keys in .env file or environment"""
    print_status("Checking for API keys...")
    
    keys_found = 0
    env_file = install_dir / ".env"
    
    # Check if .env file exists
    if env_file.exists():
        print_success(".env file found")
        try:
            with open(env_file, "r", encoding='utf-8') as f:
                content = f.read()
                
            # Simple check for actual keys (not placeholders)
            if "sk-" in content and "your-" not in content:
                keys_found += 1
                print_success("API keys appear to be configured in .env")
            else:
                print_warning(".env file exists but keys appear to be placeholders")
        except Exception:
            print_warning("Could not read .env file")
    else:
        print_warning(".env file not found")
        
        # Fallback to environment variables
        api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
        for key in api_keys:
            if os.getenv(key):
                print_success(f"{key} found in environment")
                keys_found += 1
    
    if keys_found == 0:
        print_warning("No API keys configured. Edit .env file to add your keys.")
    
    return keys_found

def test_installation(python_exe, install_dir):
    """Test the installation"""
    print_status("Testing installation...")
    
    test_script = """
try:
    print("Testing imports...")
    import aiohttp
    print("[OK] aiohttp imported")
    import yaml
    print("[OK] yaml imported")
    import rich
    print("[OK] rich imported")
    import openai
    print("[OK] openai imported")
    import anthropic
    print("[OK] anthropic imported")
    try:
        import google.generativeai
        print("[OK] google.generativeai imported")
    except ImportError:
        print("[WARN] google.generativeai not available (optional)")
    try:
        from dotenv import load_dotenv
        print("[OK] python-dotenv imported")
    except ImportError:
        print("[WARN] python-dotenv not available")
    print("[SUCCESS] All required dependencies work!")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    import sys
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    import sys
    sys.exit(1)
"""
    
    try:
        result = subprocess.run([str(python_exe), "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        print_success("Installation test passed")
        return True
    except subprocess.CalledProcessError as e:
        print_error("Installation test failed")
        print("Error output:")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        print("\nTrying individual package tests...")
        
        # Test individual packages
        packages = ["aiohttp", "yaml", "rich", "openai", "anthropic", "dotenv"]
        for package in packages:
            try:
                test_cmd = f"import {package}; print('[OK] {package} works')"
                result = subprocess.run([str(python_exe), "-c", test_cmd], 
                                      capture_output=True, text=True, check=True)
                print(result.stdout.strip())
            except subprocess.CalledProcessError as pkg_error:
                print(f"[ERROR] {package} failed: {pkg_error.stderr.strip() if pkg_error.stderr else 'Unknown error'}")
        
        return False

def download_main_app(install_dir):
    """Instructions for getting the main application"""
    print_status("Application setup...")
    
    readme_content = f"""# LLM Comparator Installation Complete!

## Next Steps:

1. **Copy the main application file:**
   - Copy `llmcompare.py` to: {install_dir}

2. **Set up API keys (EASY!):**
   - Copy `.env.template` to `.env`: 
     ```
     copy "{install_dir}/.env.template" "{install_dir}/.env"
     ```
   - Edit `.env` file with a text editor (Notepad works fine)
   - Replace the placeholder keys with your actual API keys:
     * Get OpenAI key from: https://platform.openai.com/api-keys
     * Get Anthropic key from: https://console.anthropic.com/
     * Get Google key from: https://makersuite.google.com/app/apikey

3. **Run the application:**
   
   **Windows (Multiple Options):**
   ```cmd
   # Option 1: Direct Python execution
   cd {install_dir}
   python llmcompare.py --help
   
   # Option 2: Use simple launcher
   {install_dir}\\run-llm-compare.bat --help
   
   # Option 3: Use main launcher
   {install_dir}\\llm-compare.bat --help
   ```
   
   **Mac/Linux (Multiple Options):**
   ```bash
   # Option 1: Direct Python execution
   cd {install_dir}
   python llmcompare.py --help
   
   # Option 2: Use simple launcher
   {install_dir}/run-llm-compare.sh --help
   
   # Option 3: Use main launcher
   {install_dir}/llm-compare --help
   ```

4. **Or use the activation script:**
   
   **Windows:**
   ```powershell
   . {install_dir}\\activate.ps1
   llm-compare --help
   ```
   
   **Mac/Linux:**
   ```bash
   source {install_dir}/activate.sh
   llm-compare --help
   ```

## Sample .env file:
```
OPENAI_API_KEY=sk-your-actual-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here
GOOGLE_API_KEY=your-actual-google-key-here
```

## Files Structure (UPDATED - Local Config):
```
{install_dir}/
├── llmcompare.py                  # Main app (you need to copy this)
├── .env.template                  # Template for API keys
├── .env                           # Your actual API keys (create this!)
├── .llm-compare/                  # Config and history (LOCAL!)
│   ├── config.yaml               # Configuration file
│   └── history.db                # Comparison history database
├── venv/                         # Python environment
├── logs/                         # Application logs
├── backups/                      # Config backups
├── llm-compare.bat               # Windows main launcher
├── run-llm-compare.bat           # Windows simple launcher
├── llm-compare                   # Unix main launcher  
├── run-llm-compare.sh            # Unix simple launcher
├── activate.ps1                  # Windows activation
├── activate.sh                   # Unix activation
└── README.md                     # Setup instructions
```

## Key Benefits of Local Configuration:
✅ **Portable** - Move entire folder anywhere
✅ **Self-contained** - All data stays with installation  
✅ **No user directory pollution** - Clean user profile
✅ **Team sharing** - Easy to share configured setups
✅ **Multiple installations** - Different setups in different folders

## Usage Examples:
```
python llmcompare.py "Explain quantum computing"
python llmcompare.py --template technical --topic "machine learning"  
python llmcompare.py --interactive
python llmcompare.py --help
```

## Getting Help:
- Run: python llmcompare.py --help
- Run: python llmcompare.py --templates
- Run: python llmcompare.py --interactive

## No API Keys? No Problem!
You can test with just one provider:
- OpenAI: Most popular, great quality
- Anthropic: Often very thoughtful responses
- Google: Free tier available

## Configuration Location:
Your configuration and history are stored locally in:
{install_dir}/.llm-compare/

This means:
- No files scattered in your user profile
- Easy backup (just copy the whole folder)
- Portable installations
- Multiple setups possible

## Troubleshooting:

**If launchers don't work:**
1. Try direct Python execution first:
   ```
   cd {install_dir}
   python llmcompare.py --help
   ```

2. Check if llmcompare.py exists in the installation directory

3. Try the simple launcher:
   - Windows: `run-llm-compare.bat --help`
   - Mac/Linux: `./run-llm-compare.sh --help`

**Common Issues:**
- **"File not found"**: Make sure llmcompare.py is copied to {install_dir}
- **"Permission denied"**: On Mac/Linux, run `chmod +x *.sh` in the install directory
- **"Python not found"**: The virtual environment may not be activated properly

Happy comparing! 🚀
"""
    
    with open(install_dir / "README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)
    
    print_success("Setup instructions created")

def main():
    """Main installation function"""
    print(">> LLM Comparator Simple Installer (Updated)")
    print("=" * 50)
    
    # Check requirements
    if not check_python():
        sys.exit(1)
    
    # Get paths - UPDATED to use local config
    install_dir, config_dir = get_install_paths()
    print(f"Installing to: {install_dir}")
    print(f"Config will be stored in: {config_dir}")
    
    # Setup
    create_directories(install_dir, config_dir)
    
    venv_dir = setup_virtual_environment(install_dir)
    if not venv_dir:
        sys.exit(1)
    
    python_exe = get_python_executable(venv_dir)
    
    if not install_dependencies(python_exe):
        sys.exit(1)
    
    if not create_config_file(config_dir):
        sys.exit(1)
    
    if not create_env_file(install_dir):
        sys.exit(1)
    
    create_launcher_scripts(install_dir, venv_dir)
    check_api_keys(install_dir)
    
    if not test_installation(python_exe, install_dir):
        print_warning("Installation test failed, but setup may still work")
        print("Try manually testing: python llmcompare.py --help")
    
    download_main_app(install_dir)
    
    print("\n" + "=" * 60)
    print(">> Installation Complete! (With Local Configuration)")
    print("=" * 60)
    print(f"Installation directory: {install_dir}")
    print(f"Configuration directory: {config_dir}")
    print()
    print("✅ Key Benefits:")
    print("   • Portable - move entire folder anywhere")
    print("   • Self-contained - all data stays local")
    print("   • Clean - no user directory pollution")
    print("   • Shareable - easy team distribution")
    print()
    print("Next steps:")
    print("1. Copy llmcompare.py to the installation directory")
    print("2. Copy .env.template to .env and add your API keys")
    print("3. Run the application using the launcher scripts")
    print()
    print("See README.md in the installation directory for detailed instructions")
    print("\nHappy comparing!")

if __name__ == "__main__":
    main()