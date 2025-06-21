#!/usr/bin/env python3
"""
LLM Comparator Installation Test
Quick test to verify everything is working correctly
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import aiohttp
        print("✓ aiohttp imported successfully")
    except ImportError as e:
        print(f"✗ aiohttp import failed: {e}")
        return False
    
    try:
        import yaml
        print("✓ yaml imported successfully")
    except ImportError as e:
        print(f"✗ yaml import failed: {e}")
        return False
    
    try:
        import rich
        print("✓ rich imported successfully")
    except ImportError as e:
        print(f"✗ rich import failed: {e}")
        return False
    
    try:
        import openai
        print("✓ openai imported successfully")
    except ImportError as e:
        print(f"✗ openai import failed: {e}")
        return False
    
    try:
        import anthropic
        print("✓ anthropic imported successfully")
    except ImportError as e:
        print(f"✗ anthropic import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv imported successfully")
    except ImportError as e:
        print(f"✗ python-dotenv import failed: {e}")
        return False
    
    # Optional imports
    try:
        import google.generativeai
        print("✓ google-generativeai imported successfully")
    except ImportError:
        print("⚠ google-generativeai not available (optional)")
    
    return True

def test_env_file():
    """Test .env file configuration"""
    print("\nTesting .env file configuration...")
    
    possible_env_files = [
        Path(".env"),
        Path(__file__).parent / ".env",
        Path.home() / "llm-comparator" / ".env",
    ]
    
    env_file_found = False
    for env_file in possible_env_files:
        if env_file.exists():
            print(f"✓ Found .env file at: {env_file}")
            env_file_found = True
            
            # Check if it has actual keys
            try:
                with open(env_file, 'r') as f:
                    content = f.read()
                
                if "sk-" in content and "your-" not in content:
                    print("✓ .env file appears to contain actual API keys")
                elif "your-" in content:
                    print("⚠ .env file contains placeholder keys - please add your actual API keys")
                else:
                    print("⚠ .env file may not contain valid API keys")
                    
            except Exception as e:
                print(f"⚠ Could not read .env file: {e}")
            break
    
    if not env_file_found:
        print("⚠ No .env file found")
        print("  Expected locations:")
        for env_file in possible_env_files:
            print(f"    {env_file}")
    
    return env_file_found

def test_llm_comparator():
    """Test that the main LLM Comparator can be imported and initialized"""
    print("\nTesting LLM Comparator initialization...")
    
    try:
        # Try to import the main module
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Try different possible filenames
        possible_names = ['llm_comparator', 'llmcompare', 'main']
        
        module = None
        for name in possible_names:
            try:
                module = __import__(name)
                print(f"✓ Successfully imported {name}")
                break
            except ImportError:
                continue
        
        if not module:
            print("✗ Could not import LLM Comparator module")
            print("  Make sure llm_comparator.py is in the current directory")
            return False
        
        # Try to get the main class
        if hasattr(module, 'EnhancedLLMComparator'):
            comparator_class = module.EnhancedLLMComparator
        elif hasattr(module, 'LLMComparator'):
            comparator_class = module.LLMComparator
        else:
            print("✗ Could not find LLMComparator class in module")
            return False
        
        # Try to initialize without args
        try:
            comparator = comparator_class()
            print("✓ LLM Comparator initialized successfully")
            
            # Test basic functionality
            if hasattr(comparator, 'clients'):
                if comparator.clients:
                    print(f"✓ Found {len(comparator.clients)} configured API client(s)")
                else:
                    print("⚠ No API clients configured (check your .env file)")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize LLM Comparator: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing LLM Comparator: {e}")
        return False

def test_api_keys():
    """Test API key configuration"""
    print("\nTesting API key configuration...")
    
    # Load .env file if available
    try:
        from dotenv import load_dotenv
        
        possible_env_files = [
            Path(".env"),
            Path(__file__).parent / ".env",
            Path.home() / "llm-comparator" / ".env",
        ]
        
        for env_file in possible_env_files:
            if env_file.exists():
                load_dotenv(env_file)
                break
                
    except ImportError:
        print("⚠ python-dotenv not available, checking system environment only")
    
    # Check for API keys
    api_keys = {
        "OPENAI_API_KEY": "OpenAI (GPT models)",
        "ANTHROPIC_API_KEY": "Anthropic (Claude models)",
        "GOOGLE_API_KEY": "Google (Gemini models)"
    }
    
    configured_keys = 0
    for key, description in api_keys.items():
        value = os.getenv(key)
        if value and value.startswith(('sk-', 'AIza')) and 'your-' not in value:
            print(f"✓ {description}: Configured")
            configured_keys += 1
        else:
            print(f"⚠ {description}: Not configured")
    
    if configured_keys == 0:
        print("⚠ No API keys configured")
        print("  Please add your API keys to the .env file")
        return False
    else:
        print(f"✓ {configured_keys} API key(s) configured")
        return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("LLM Comparator Installation Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Package imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: .env file
    if test_env_file():
        tests_passed += 1
    
    # Test 3: API keys
    if test_api_keys():
        tests_passed += 1
    
    # Test 4: LLM Comparator
    if test_llm_comparator():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    print("=" * 50)
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your installation is working correctly.")
        print("\nYou can now run:")
        print("  llm-compare --help")
        print("  llm-compare --config")
        print("  llm-compare \"Your first prompt here\"")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        if tests_passed >= 2:
            print("Your installation is partially working - you may still be able to use the tool.")
        
        print("\nFor help:")
        print("  llm-compare --setup-help")
        print("  llm-compare --config")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)