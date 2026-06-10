# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq Provider Setup ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")


# --- Data Path Configurations ---
TEST_CASES_PATH = os.getenv("TEST_CASES_PATH", "data/sample_inputs/test_cases.json")
RESULTS_JSON_PATH = os.getenv("RESULTS_JSON_PATH", "data/sample_outputs/results.json")
RESULTS_CSV_PATH = os.getenv("RESULTS_CSV_PATH", "data/sample_outputs/results.csv")

# --- Startup Validations ---
if not GROQ_API_KEY:
    raise ValueError("CRITICAL ERROR: GROQ_API_KEY is not set in your .env file.")
