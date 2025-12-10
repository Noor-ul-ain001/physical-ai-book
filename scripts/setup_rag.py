#!/usr/bin/env python3
"""
Interactive RAG Setup Script
Helps configure environment variables for the RAG chatbot
"""

import os
from pathlib import Path
import sys

def print_header():
    print("\n" + "="*70)
    print("🤖  Physical AI Textbook - RAG Chatbot Setup")
    print("="*70 + "\n")

def print_section(title):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print("─"*70 + "\n")

def get_input(prompt, default=""):
    """Get user input with optional default value"""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        while True:
            user_input = input(f"{prompt}: ").strip()
            if user_input:
                return user_input
            print("  ⚠️  This field is required. Please enter a value.")

def check_existing_env():
    """Check if .env file exists and has values"""
    env_path = Path(__file__).parent.parent / ".env"

    if not env_path.exists():
        return None

    existing_config = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                existing_config[key] = value

    return existing_config

def main():
    print_header()

    # Check existing configuration
    existing_config = check_existing_env()

    if existing_config:
        print("✓ Found existing .env file\n")
        print("Current configuration:")
        for key in ['GEMINI_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']:
            value = existing_config.get(key, 'Not set')
            masked_value = value if 'your' in value or not value or value == 'Not set' else value[:10] + '...'
            print(f"  {key}: {masked_value}")

        print("\nDo you want to update the configuration?")
        update = input("  Enter 'y' to update, or any other key to skip: ").lower()

        if update != 'y':
            print("\n✓ Keeping existing configuration")
            return
    else:
        print("📝 No .env file found. Let's create one!\n")

    # Collect configuration
    print_section("1. Gemini API Configuration")
    print("Get your API key from: https://makersuite.google.com/app/apikey\n")

    gemini_key = get_input(
        "Enter your Gemini API key",
        existing_config.get('GEMINI_API_KEY', '') if existing_config else ''
    )

    print_section("2. Qdrant Cloud Configuration")
    print("Sign up at: https://cloud.qdrant.io/")
    print("Create a cluster and get your URL and API key\n")

    qdrant_url = get_input(
        "Enter your Qdrant cluster URL (e.g., https://xyz.qdrant.tech)",
        existing_config.get('QDRANT_URL', '') if existing_config else ''
    )

    qdrant_key = get_input(
        "Enter your Qdrant API key",
        existing_config.get('QDRANT_API_KEY', '') if existing_config else ''
    )

    print_section("3. Optional Configuration")
    print("You can skip these for now (press Enter to use defaults)\n")

    neon_db = get_input(
        "Neon Database URL (optional)",
        existing_config.get('NEON_DB_URL', 'postgresql://user:password@host/dbname') if existing_config else 'postgresql://user:password@host/dbname'
    )

    better_auth = get_input(
        "Better Auth Secret (optional)",
        existing_config.get('BETTER_AUTH_SECRET', 'your_secret_here') if existing_config else 'your_secret_here'
    )

    # Write .env file
    print_section("Saving Configuration")

    env_path = Path(__file__).parent.parent / ".env"

    env_content = f"""# Environment variables for Physical AI & Humanoid Robotics project

# Gemini API Configuration
GEMINI_API_KEY={gemini_key}

# Qdrant Configuration
QDRANT_URL={qdrant_url}
QDRANT_API_KEY={qdrant_key}

# Database Configuration
NEON_DB_URL={neon_db}

# Auth Configuration
BETTER_AUTH_SECRET={better_auth}

# Isaac Sim Configuration
ISAAC_SIM_PATH=/path/to/isaac/sim
"""

    with open(env_path, 'w') as f:
        f.write(env_content)

    print(f"✓ Configuration saved to: {env_path}\n")

    # Next steps
    print_section("Next Steps")
    print("""
1. Index your content:
   python scripts/index_content.py

2. Start the RAG-enabled backend:
   cd api
   python rag_main.py

3. Start the frontend (in another terminal):
   npm start

4. Open http://localhost:3000 and test the chatbot!

📚 For detailed instructions, see: RAG_SETUP.md
    """)

    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}\n")
        sys.exit(1)
