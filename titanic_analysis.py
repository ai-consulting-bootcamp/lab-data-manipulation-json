"""
Titanic Data Analysis and JSON Export
Author: [Your Name]
Description: Analyze Titanic passenger data, engineer features, and export to JSON
"""
 
import pandas as pd
import numpy as np
import json
from pathlib import Path
 
# Set up paths
DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "titanic.csv"
JSON_FILE = DATA_DIR / "titanic_data.json"
 
# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)
 
print("Project setup complete!")
print(f"Data directory: {DATA_DIR}")
print(f"CSV file location: {CSV_FILE}")
