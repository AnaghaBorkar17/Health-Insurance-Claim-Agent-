"""Phase 1 Verification Script.

Tests:
1. Python version (>= 3.11)
2. Virtual environment presence
3. Local Ollama server connectivity and model availability (qwen2.5:3b, nomic-embed-text)
4. Environment variables and Google Gemini API key status
"""

import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

# Color styling for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_check(name: str, passed: bool, details: str = ""):
    status = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
    print(f"  {status} {name}")
    if details:
        indent = " " * 9
        print(f"{indent}{details}")


def print_warn(name: str, details: str = ""):
    print(f"  {YELLOW}[WARN]{RESET} {name}")
    if details:
        indent = " " * 9
        print(f"{indent}{details}")


def check_python_version() -> bool:
    print(f"\n{BLUE}[1/4] Checking Python Environment...{RESET}")
    v = sys.version_info
    passed = v.major == 3 and v.minor >= 11
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    print_check(
        f"Python Version: {version_str}",
        passed,
        "Compatible with Python 3.11+" if passed else "Please use Python 3.11 or newer",
    )

    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print_check("Virtual Environment: Active", True, f"Prefix: {sys.prefix}")
    else:
        print_warn(
            "Virtual Environment: Not Active",
            "Recommended to run inside 'venv'. Activate using: .\\venv\\Scripts\\Activate.ps1",
        )
    return passed


def check_ollama() -> bool:
    print(f"\n{BLUE}[2/4] Checking Ollama Local Runtime & Models...{RESET}")
    ollama_url = "http://127.0.0.1:11434/api/tags"
    try:
        req = urllib.request.Request(ollama_url)
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                print_check("Ollama Local Server: Reachable", True, f"Endpoint: {ollama_url}")

                has_llm = any("qwen2.5" in m for m in models)
                has_embed = any("nomic-embed-text" in m for m in models)

                print_check(
                    "Ollama LLM (qwen2.5:3b)",
                    has_llm,
                    f"Found installed: {[m for m in models if 'qwen' in m]}" if has_llm else "Run: ollama pull qwen2.5:3b",
                )
                print_check(
                    "Ollama Embedding (nomic-embed-text)",
                    has_embed,
                    f"Found installed: {[m for m in models if 'nomic' in m]}" if has_embed else "Run: ollama pull nomic-embed-text",
                )
                return has_llm and has_embed
            else:
                print_check("Ollama Local Server", False, f"HTTP Status: {response.status}")
                return False
    except urllib.error.URLError as e:
        print_check("Ollama Local Server: Not Reachable", False, f"Error: {e}. Ensure Ollama is running.")
        return False


def check_configuration() -> bool:
    print(f"\n{BLUE}[3/4] Checking Configuration & .env File...{RESET}")
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print_check(".env file presence", False, f"Missing {env_path}. Copy from .env.example")
        return False

    print_check(".env file presence", True, f"Located at: {env_path}")

    # Read .env file without external dependencies
    env_vars = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()

    api_key = env_vars.get("GEMINI_API_KEY") or env_vars.get("GOOGLE_API_KEY", "")
    is_placeholder = not api_key or api_key in {
        "your_gemini_api_key_here",
        "your_api_key",
        "replace_me",
    }

    if is_placeholder:
        print_warn(
            "GEMINI_API_KEY / GOOGLE_API_KEY: Not yet configured",
            "Using placeholder. You can obtain a free key at https://aistudio.google.com/ and paste it into .env",
        )
        return False
    else:
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        print_check("API Key (GEMINI_API_KEY / GOOGLE_API_KEY): Configured", True, f"Key set ({masked_key})")
        return True


def check_gemini_api() -> None:
    print(f"\n{BLUE}[4/4] Optional Gemini API Connectivity Ping...{RESET}")
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return

    env_vars = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()

    api_key = env_vars.get("GEMINI_API_KEY") or env_vars.get("GOOGLE_API_KEY", "")
    if not api_key or api_key in {"your_gemini_api_key_here", "replace_me"}:
        print("  [SKIP] Skipping live Gemini call until API key is provided.")
        return

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=15000))
        model_name = env_vars.get("GEMINI_MODEL", "gemini-3.8-flash")
        response = client.models.generate_content(
            model=model_name,
            contents="Respond with only the word 'READY'",
        )
        text = response.text.strip()
        print_check("Google Gemini Live Ping", True, f"Model responded: {text}")
    except ImportError:
        print_warn("google-genai package not yet installed in active environment.")
    except Exception as e:
        print_check("Google Gemini Live Ping", False, f"API Call Notice: {e}")


def main():
    print("=" * 60)
    print(" CLAIM AGENTS: PHASE 1 ENVIRONMENT & SETUP VERIFICATION")
    print("=" * 60)

    py_ok = check_python_version()
    ollama_ok = check_ollama()
    env_ok = check_configuration()
    check_gemini_api()

    print("\n" + "=" * 60)
    if py_ok and ollama_ok:
        print(f"{GREEN}[OK] Phase 1 Core Infrastructure is Ready!{RESET}")
        if not env_ok:
            print(f"{YELLOW}Note: Add your GEMINI_API_KEY to claim-agents/.env when ready.{RESET}")
    else:
        print(f"{RED}[FAIL] Please address the failed checks above before continuing.{RESET}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
