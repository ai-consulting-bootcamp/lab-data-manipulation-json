"""
Titanic Data Manipulation and JSON Handling
Author: Gretel Ceballos
Description: Analyze Titanic passenger data, engineer new features,
and export the processed dataset to JSON format.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# Set up project paths
DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "titanic.csv"
JSON_FILE = DATA_DIR / "titanic_processed.json"


# Create the data directory if it does not exist
DATA_DIR.mkdir(exist_ok=True)


print("=" * 60)
print("TITANIC DATA MANIPULATION AND JSON HANDLING")
print("=" * 60)
print("Project setup complete!")
print(f"Data directory: {DATA_DIR}")
print(f"CSV file location: {CSV_FILE}")
print(f"JSON file location: {JSON_FILE}")

# ============================
# Load the Titanic dataset
# ============================

print("Loading dataset...\n")

df = pd.read_csv(CSV_FILE)

print("Dataset loaded successfully!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

# ============================
# Descriptive statistics
# ============================

print("\n" + "=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)

# Select numeric columns only
numeric_columns = df.select_dtypes(include="number").columns

print(f"\nNumeric columns: {list(numeric_columns)}")

# Calculate mean, median, and standard deviation
statistics_data = {}
for column in numeric_columns:
    mean_value = df[column].mean()
    median_value = df[column].median()
    std_value = df[column].std()

    statistics_data[column] = {
        "mean": round(float(mean_value), 2),
        "median": round(float(median_value), 2),
        "standard_deviation": round(float(std_value), 2)
    }

    print(f"\n{column}:")
    print(f"  Mean: {mean_value:.2f}")
    print(f"  Median: {median_value:.2f}")
    print(f"  Standard deviation: {std_value:.2f}")

# ============================
# Missing values analysis
# ============================

print("\n" + "=" * 50)
print("MISSING VALUES ANALYSIS")
print("=" * 50)

missing_data = {}

for col in df.columns:
    missing_count = df[col].isna().sum()
    missing_percent = (missing_count / len(df)) * 100

    missing_data[col] = {
        "missing_count": int(missing_count),
        "missing_percent": round(missing_percent, 2)
    }

    print(
        f"{col}: "
        f"{missing_count} missing values "
        f"({missing_percent:.2f}%)"
    )

# ==================================================
# FEATURE ENGINEERING
# ==================================================

# Create a copy of the dataframe for feature engineering
df_features = df.copy()

# Feature 1: Family Size
df_features['FamilySize'] = df_features['SibSp'] + df_features['Parch'] + 1

print("\nFirst 10 rows - Family Size")
print(df_features[['SibSp', 'Parch', 'FamilySize']].head(10))

# Feature 2: Is Alone
df_features['IsAlone'] = df_features['FamilySize'] == 1

print("\nFirst 10 rows - Is Alone")
print(df_features[['FamilySize', 'IsAlone']].head(10))

# Feature 3: Age Groups
def categorize_age(age):
    """Categorize age into groups"""
    if pd.isna(age):
        return "Unknown"
    elif age < 18:
        return "Child"
    elif age < 30:
        return "Young Adult"
    elif age < 50:
        return "Adult"
    else:
        return "Senior"

df_features["AgeGroup"] = df_features["Age"].apply(categorize_age)

print("\nFirst 10 rows - Age Groups")
print(df_features[["Age", "AgeGroup"]].head(10))

# Analyze feature differences between survivors and non-survivors
print("\n" + "=" * 50)
print("FEATURE ANALYSIS: SURVIVED vs NOT SURVIVED")
print("=" * 50)

print("\nFamily Size by Survival:")
family_survival = df_features.groupby("Survived")["FamilySize"].agg(["mean", "median", "std"])
print(family_survival)

# Statistical comparison
print("\n" + "=" * 50)
print("FEATURE DIFFERENTIATION ANALYSIS")
print("=" * 50)

survived = df_features[df_features["Survived"] == 1]
not_survived = df_features[df_features["Survived"] == 0]

print("\nFamily Size:")
print(f"  Survived mean: {survived['FamilySize'].mean():.2f}")
print(f"  Not Survived mean: {not_survived['FamilySize'].mean():.2f}")
print(f"  Difference: {abs(survived['FamilySize'].mean() - not_survived['FamilySize'].mean()):.2f}")

class Passenger:
    """
    Represents a passenger with all their information.
    """

    def __init__(self, passenger_id, name, age, sex, survived, pclass,
                 fare, embarked=None, family_size=None, is_alone=None, age_group=None):

        self.passenger_id = int(passenger_id) if pd.notna(passenger_id) else None
        self.name = str(name) if pd.notna(name) else None
        self.age = float(age) if pd.notna(age) else None
        self.sex = str(sex) if pd.notna(sex) else None
        self.survived = int(survived) if pd.notna(survived) else None
        self.pclass = int(pclass) if pd.notna(pclass) else None
        self.fare = float(fare) if pd.notna(fare) else None
        self.embarked = str(embarked) if pd.notna(embarked) else None
        self.family_size = int(family_size) if pd.notna(family_size) else None
        self.is_alone = bool(is_alone) if pd.notna(is_alone) else None
        self.age_group = str(age_group) if pd.notna(age_group) else None

    def to_dict(self):
        """Convert passenger to dictionary for JSON serialization."""
        return {
            "passenger_id": self.passenger_id,
            "name": self.name,
            "age": self.age,
            "sex": self.sex,
            "survived": self.survived,
            "pclass": self.pclass,
            "fare": self.fare,
            "embarked": self.embarked,
            "family_size": self.family_size,
            "is_alone": self.is_alone,
            "age_group": self.age_group
        }
    
class TitanicDataset:
    """
    Manages the Titanic passenger dataset.
    """

    def __init__(self):
        self.passengers = []

    def load_from_dataframe(self, dataframe):
        """Load passengers from a pandas DataFrame."""
        for _, row in dataframe.iterrows():
            passenger = Passenger(
                passenger_id=row["PassengerId"],
                name=row["Name"],
                age=row["Age"],
                sex=row["Sex"],
                survived=row["Survived"],
                pclass=row["Pclass"],
                fare=row["Fare"],
                embarked=row["Embarked"],
                family_size=row["FamilySize"],
                is_alone=row["IsAlone"],
                age_group=row["AgeGroup"]
            )
            self.passengers.append(passenger)

# ============================
# Create Passenger objects
# ============================

dataset = TitanicDataset()
dataset.load_from_dataframe(df_features)

print("\n" + "=" * 50)
print("OBJECT CREATION")
print("=" * 50)
print(f"Passenger objects created: {len(dataset.passengers)}")

# ============================
# Export processed data to JSON
# ============================

output_data = {
    "metadata": {
        "generated_at": datetime.now().isoformat(),
        "total_passengers": len(dataset.passengers),
        "source_file": str(CSV_FILE)
    },
    "statistics": statistics_data,
    "missing_values": missing_data,
    "passengers": [
        passenger.to_dict()
        for passenger in dataset.passengers
    ]
}

with open(JSON_FILE, "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, indent=4, ensure_ascii=False)

print("\n" + "=" * 50)
print("JSON EXPORT")
print("=" * 50)
print(f"JSON file created successfully: {JSON_FILE}")

# ============================
# JSON VALIDATION
# ============================

print("\n" + "=" * 50)
print("JSON VALIDATION")
print("=" * 50)

with open(JSON_FILE, "r", encoding="utf-8") as json_file:
    json_data = json.load(json_file)

print("JSON loaded successfully!")
print(f"Metadata keys: {list(json_data['metadata'].keys())}")
print(f"Statistics keys: {list(json_data['statistics'].keys())}")
print(f"Missing values keys: {list(json_data['missing_values'].keys())}")
print(f"Passenger records: {len(json_data['passengers'])}")