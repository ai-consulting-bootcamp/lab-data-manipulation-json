# %% titanic_analysis_export.py
"""
Titanic Data Analysis and JSON Export
Author: [Bhabishwor]
Description: Analyze Titanic passenger data, engineer features, and export to JSON
"""
 
import pandas as pd
import numpy as np
import json
from pathlib import Path
 
# Set up paths relative to this script, so it works from any terminal folder.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_FILE = BASE_DIR / "Titanic.csv"
JSON_FILE = DATA_DIR / "titanic_data.json"
EVIDENCE_FILE = BASE_DIR / "titanic_submission_output.txt"
 
# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)
 
print("Project setup complete!")
print(f"Data directory: {DATA_DIR}")
print(f"CSV file location: {CSV_FILE}")

# %%
df = pd.read_csv(CSV_FILE)
print(f"Dataset loaded successfully! Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# %%
#step 3: calculating descriptive statistics for numerical columns

print("\n" + "=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)

# Select numeric columns
numeric_columns = df.select_dtypes(include="number")

print("\nNumeric columns:")
print(list(numeric_columns.columns))

# Calculate statistics
means = numeric_columns.mean()
medians = numeric_columns.median()
standard_deviations = numeric_columns.std()

print("\nMeans:")
print(means)

print("\nMedians:")
print(medians)

print("\nStandard deviations:")
print(standard_deviations)

# %%
# Step 4: Identifying missing values

print("\n" + "=" * 50)
print("MISSING VALUES ANALYSIS")
print("=" * 50)

missing_data = {}

for col in df.columns:
    missing_count = df[col].isna().sum()
    missing_percent = (missing_count / len(df)) * 100

    missing_data[col] = {
        "count": missing_count,
        "percentage": missing_percent
    }

    print(f"{col}: {missing_count} missing values "
          f"({missing_percent:.2f}%)")

# %%
# Step 5: Feature Engineering

print("\n" + "=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)

# Work on a copy so the original DataFrame remains unchanged
df_features = df.copy()

# Feature 1: FamilySize
df_features["FamilySize"] = (
    df_features["SibSp"] + df_features["Parch"] + 1
)

print("\nFamilySize:")
print(df_features[["SibSp", "Parch", "FamilySize"]].head(10))

# %%
# Feature 2: IsAlone
df_features["IsAlone"] = df_features["FamilySize"] == 1

print("\nIsAlone:")
print(df_features[["FamilySize", "IsAlone"]].head(10))

# %%
# Feature 3: AgeGroup

def categorize_age(age):
    """Place an age into an age category."""
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

print("\nAgeGroup:")
print(df_features[["Age", "AgeGroup"]].head(10))

# %%
#Analyze feature differences between survivors and non-survivors
#but first separate the data into two groups based on survival status

#  survived = df_features[df_features["Survived"] == 1]
not_survived = df_features[df_features["Survived"] == 0]

# %%
#then comparing family size between survivors and non-survivors
print("\nFamily size by survival status:")

family_survival = df_features.groupby("Survived")["FamilySize"].agg(
    ["mean", "median", "std"]
)

print(family_survival)

# %%
# Separate the two survival groups
survived = df_features[df_features["Survived"] == 1]
not_survived = df_features[df_features["Survived"] == 0]

# %%
#comparing the average family size directly
print(f"\nSurvived family-size mean: "
      f"{survived['FamilySize'].mean():.2f}")

print(f"Did not survive family-size mean: "
      f"{not_survived['FamilySize'].mean():.2f}")

# %%
#Finally, examine the age groups
print("\nSurvival rate by age group:")

age_group_survival = df_features.groupby("AgeGroup")["Survived"].mean()
print(age_group_survival)

# %%
print("\nSurvival counts by age group:")
print(pd.crosstab(df_features["AgeGroup"], df_features["Survived"]))

# %%
print(df_features[["FamilySize", "IsAlone", "AgeGroup"]].head())

# %%
# Step 6: Classes for JSON Export

class Passenger:
    """Represents one Titanic passenger."""

    def __init__(
        self,
        passenger_id,
        name,
        age,
        sex,
        survived,
        pclass,
        fare,
        embarked=None,
        family_size=None,
        is_alone=None,
        title=None
    ):
        # Convert missing values to None
        self.passenger_id = (
            int(passenger_id) if pd.notna(passenger_id) else None
        )

        self.name = name if pd.notna(name) else None
        self.age = float(age) if pd.notna(age) else None
        self.sex = sex if pd.notna(sex) else None
        self.survived = int(survived) if pd.notna(survived) else None
        self.pclass = int(pclass) if pd.notna(pclass) else None
        self.fare = float(fare) if pd.notna(fare) else None
        self.embarked = embarked if pd.notna(embarked) else None

        self.family_size = (
            int(family_size) if pd.notna(family_size) else None
        )

        self.is_alone = (
            bool(is_alone) if pd.notna(is_alone) else None
        )

        self.title = title if pd.notna(title) else None

    def to_dict(self):
        """Convert this passenger into a JSON-compatible dictionary."""
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
            "title": self.title
        }

# %%
#Testing the class with a sample passenger just one row from the DataFrame
first_row = df_features.iloc[0]

passenger = Passenger(
    passenger_id=first_row["PassengerId"],
    name=first_row["Name"],
    age=first_row["Age"],
    sex=first_row["Sex"],
    survived=first_row["Survived"],
    pclass=first_row["Pclass"],
    fare=first_row["Fare"],
    embarked=first_row["Embarked"],
    family_size=first_row["FamilySize"],
    is_alone=first_row["IsAlone"]
)

print(passenger.to_dict())

# %%
first_row = df_features.iloc[0]

test_passenger = Passenger(
    passenger_id=first_row["PassengerId"],
    name=first_row["Name"],
    age=first_row["Age"],
    sex=first_row["Sex"],
    survived=first_row["Survived"],
    pclass=first_row["Pclass"],
    fare=first_row["Fare"],
    embarked=first_row["Embarked"],
    family_size=first_row["FamilySize"],
    is_alone=first_row["IsAlone"]
)

print(test_passenger.to_dict())

# %%
class TitanicDataset:
    """Manages the processed Titanic dataset."""

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.passengers = []

    def create_passengers(self):
        """Create Passenger objects from dataframe rows."""
        # TODO: loop through self.dataframe.iterrows()
        # TODO: create one Passenger object per row
        # TODO: append each object to self.passengers
        pass

    def to_dict(self):
        """Return the dataset in a JSON-compatible structure."""
        # TODO: convert each Passenger object using passenger.to_dict()
        pass

    def export_to_json(self, file_path):
        """Export the dataset to a JSON file."""
        # TODO: call create_passengers()
        # TODO: open file_path for writing
        # TODO: use json.dump(...)
        pass

# %%
dataset = TitanicDataset(df_features)

# %%
print([name for name in dir(dataset) if not name.startswith("_")])

# %%
# Step 7: Check which parts currently exist

required_names = [
    "df",
    "df_features",
    "Passenger",
    "TitanicDataset",
    "JSON_FILE"
]

for name in required_names:
    if name in globals():
        print(f"✓ {name} is available")
    else:
        print(f"✗ {name} is missing")

# %%
# Validate data loading
print("Dataset shape:", df.shape)
assert not df.empty

# Validate engineered features
for feature in ["FamilySize", "IsAlone", "AgeGroup"]:
    assert feature in df_features.columns
    print(f"✓ {feature} exists")

# Validate missing-value analysis
missing_counts = df.isna().sum()
print("\nMissing values:")
print(missing_counts)

# Validate Passenger class
first_row = df_features.iloc[0]

test_passenger = Passenger(
    passenger_id=first_row["PassengerId"],
    name=first_row["Name"],
    age=first_row["Age"],
    sex=first_row["Sex"],
    survived=first_row["Survived"],
    pclass=first_row["Pclass"],
    fare=first_row["Fare"],
    embarked=first_row["Embarked"],
    family_size=first_row["FamilySize"],
    is_alone=first_row["IsAlone"]
)

print("\nPassenger dictionary:")
print(test_passenger.to_dict())

# %%
print(JSON_FILE.resolve())
print(JSON_FILE.exists())

# %%
passenger_records = []

for _, row in df_features.iterrows():
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
        is_alone=row["IsAlone"]
    )

    passenger_records.append(passenger.to_dict())

# %%
DATA_DIR.mkdir(exist_ok=True)

with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(passenger_records, f, indent=2)

print(f"JSON file created at: {JSON_FILE.resolve()}")

# %%
print(JSON_FILE.exists())

# %%
with open(JSON_FILE, "r", encoding="utf-8") as f:
    json_data = json.load(f)

print("JSON validation passed")
print(f"Records loaded: {len(json_data)}")

# %%
JSON_FILE = DATA_DIR / "titanic_data.json"

# %%
# Step 7: Testing and Validation

with open(JSON_FILE, "r", encoding="utf-8") as f:
    json_data = json.load(f)

print("JSON validation passed")
print(f"Number of records: {len(json_data)}")
print(f"Data type: {type(json_data).__name__}")

# %%
# Confirm that JSON contains records
assert isinstance(json_data, list)
assert len(json_data) == len(df_features)

# Confirm that the first record is a dictionary
assert isinstance(json_data[0], dict)

# Confirm expected fields exist
expected_fields = [
    "passenger_id",
    "name",
    "age",
    "sex",
    "survived",
    "pclass",
    "fare",
    "embarked",
    "family_size",
    "is_alone",
    "title"
]

for field in expected_fields:
    assert field in json_data[0], f"Missing field: {field}"

print("JSON structure validation passed")
print("\nFirst passenger record:")
print(json_data[0])

# %%
# Create a concise text artifact for assignment submission.
statistics_table = pd.DataFrame({
    "Mean": means,
    "Median": medians,
    "Standard deviation": standard_deviations
}).round(2)

missing_table = pd.DataFrame.from_dict(missing_data, orient="index")
missing_table["percentage"] = missing_table["percentage"].map(
    lambda value: f"{value:.2f}%"
)
missing_table.columns = ["Missing count", "Missing percentage"]

evidence = "\n".join([
    "TITANIC ANALYSIS - SUBMISSION EVIDENCE",
    "=" * 48,
    f"Dataset: {CSV_FILE.name} ({df.shape[0]} rows, {df.shape[1]} columns)",
    "",
    "1. Statistics calculations (mean, median, standard deviation)",
    statistics_table.to_string(),
    "",
    "2. Missing value analysis",
    missing_table.to_string(),
    "",
    "3. Feature engineering results (first 10 rows)",
    df_features[["FamilySize", "IsAlone", "AgeGroup"]]
        .head(10).to_string(index=False),
    "",
    "4. JSON export confirmation",
    f"File: {JSON_FILE.resolve()}",
    f"File exists: {JSON_FILE.exists()}",
    f"Records loaded and validated: {len(json_data)}",
    f"Expected records: {len(df_features)}",
    "JSON structure validation: passed"
])

EVIDENCE_FILE.write_text(evidence + "\n", encoding="utf-8")
print("\n" + evidence)
print(f"\nSubmission text saved to: {EVIDENCE_FILE.resolve()}")

# %%



