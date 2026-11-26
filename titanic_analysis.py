"""
Titanic Data Analysis and JSON Export
Author: Sandra
Description: Analyse passenger data and export results to JSON.
"""

from pathlib import Path

import pandas as pd
import json
from datetime import datetime, timezone

# Build paths from this script's location.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_FILE = DATA_DIR / "titanic.csv"
JSON_FILE = DATA_DIR / "titanic_processed.json"

# Load the original dataset without changing it.
df = pd.read_csv(CSV_FILE)

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print("\nFirst five rows:")
print(df.head())

# Calculate descriptive statistics for all numeric columns.
numeric_data = df.select_dtypes(include="number")

descriptive_stats = pd.DataFrame({
    "mean": numeric_data.mean(),
    "median": numeric_data.median(),
    "std": numeric_data.std(ddof=1),
})

print("\nDESCRIPTIVE STATISTICS")
print(descriptive_stats.round(2))

# Count missing values and calculate percentages.
missing_data = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": df.isna().mean() * 100,
}).sort_values("missing_count", ascending=False)

print("\nMISSING VALUES ANALYSIS")
print(missing_data.round(2))

# Create new features while keeping the original dataframe unchanged.
df_features = df.copy()

df_features["FamilySize"] = (
    df_features["SibSp"] + df_features["Parch"] + 1
)
df_features["IsAlone"] = df_features["FamilySize"] == 1


def categorize_age(age):
    """Return an age group, keeping missing ages as Unknown."""
    if pd.isna(age):
        return "Unknown"
    elif age < 18:
        return "Under 18"
    elif age < 30:
        return "18-29"
    elif age < 50:
        return "30-49"
    else:
        return "50+"


df_features["AgeGroup"] = df_features["Age"].apply(categorize_age)

print("\nENGINEERED FEATURES")
print(
    df_features[
        ["Age", "SibSp", "Parch", "FamilySize", "IsAlone", "AgeGroup"]
    ].head(10)
)

print("\nAGE GROUP COUNTS")
print(df_features["AgeGroup"].value_counts())

# Compare family size between survival groups.
family_by_survival = (
    df_features.groupby("Survived")["FamilySize"]
    .agg(["count", "mean", "median", "std"])
)

print("\nFAMILY SIZE BY SURVIVAL (0 = No, 1 = Yes)")
print(family_by_survival.round(2))

# Compare survival rates by each engineered feature.
feature_analysis = {}

for feature in ["FamilySize", "IsAlone", "AgeGroup"]:
    comparison = df_features.groupby(feature)["Survived"].agg(
        passengers="count",
        survived="sum",
        survival_rate="mean",
    )

    # Keep rates as proportions for later JSON export.
    feature_analysis[feature] = comparison

    display_table = comparison.copy()
    display_table["survival_percent"] = (
        display_table["survival_rate"] * 100
    )

    print(f"\nSURVIVAL BY {feature}")
    print(
        display_table[
            ["passengers", "survived", "survival_percent"]
        ].round(2)
    )

class Passenger:
    """Represent one passenger with original and engineered data."""

    def __init__(self, row):
        integer_fields = {
            "passenger_id": "PassengerId",
            "survived": "Survived",
            "pclass": "Pclass",
            "sibsp": "SibSp",
            "parch": "Parch",
            "family_size": "FamilySize",
        }

        float_fields = {
            "age": "Age",
            "fare": "Fare",
        }

        text_fields = {
            "name": "Name",
            "sex": "Sex",
            "ticket": "Ticket",
            "cabin": "Cabin",
            "embarked": "Embarked",
            "age_group": "AgeGroup",
        }

        for attribute, column in integer_fields.items():
            value = row[column]
            setattr(self, attribute, int(value) if pd.notna(value) else None)

        for attribute, column in float_fields.items():
            value = row[column]
            setattr(self, attribute, float(value) if pd.notna(value) else None)

        for attribute, column in text_fields.items():
            value = row[column]
            setattr(self, attribute, str(value) if pd.notna(value) else None)

        value = row["IsAlone"]
        self.is_alone = bool(value) if pd.notna(value) else None

    def to_dict(self):
        """Return passenger attributes as a dictionary."""
        return vars(self).copy()


# Check a passenger whose age is missing.
sample_passenger = Passenger(df_features.iloc[5])

print("\nPASSENGER CLASS CHECK")
print(sample_passenger.to_dict())

class TitanicDataset:
    """Store passenger objects and dataset analysis."""

    def __init__(
        self,
        dataframe,
        descriptive_stats,
        missing_data,
        feature_analysis,
        family_by_survival,
    ):
        self.dataframe = dataframe.copy()
        self.descriptive_stats = descriptive_stats.copy()
        self.missing_data = missing_data.copy()
        self.feature_analysis = {
            name: table.copy()
            for name, table in feature_analysis.items()
        }
        self.family_by_survival = family_by_survival.copy()
        self.passengers = []
        self._create_passengers()

    def _create_passengers(self):
        """Create one Passenger object for each dataframe row."""
        for _, row in self.dataframe.iterrows():
            self.passengers.append(Passenger(row))

    def get_summary_stats(self):
        """Return counts and averages, excluding missing values."""
        survived_count = sum(
            passenger.survived == 1
            for passenger in self.passengers
        )
        not_survived_count = sum(
            passenger.survived == 0
            for passenger in self.passengers
        )

        ages = [
            passenger.age
            for passenger in self.passengers
            if passenger.age is not None
        ]
        fares = [
            passenger.fare
            for passenger in self.passengers
            if passenger.fare is not None
        ]

        total = len(self.passengers)

        return {
            "total_passengers": total,
            "survived": survived_count,
            "did_not_survive": not_survived_count,
            "survival_rate": survived_count / total if total else None,
            "average_age": sum(ages) / len(ages) if ages else None,
            "average_fare": sum(fares) / len(fares) if fares else None,
        }
    @staticmethod
    def _table_to_records(table):
        """Convert a table to records using native JSON types."""
        return json.loads(
            table.reset_index().to_json(orient="records")
        )

    def to_json(self, filename):
        """Export passengers, metadata and analysis to JSON."""
        data = {
            "metadata": {
                "dataset_name": "Titanic Passenger Dataset",
                "source": "titanic.csv supplied by the instructor",
                "export_date": datetime.now(timezone.utc).isoformat(),
                "total_passengers": len(self.passengers),
                "engineered_features": [
                    "FamilySize", "IsAlone", "AgeGroup"
                ],
                "missing_value_policy": (
                    "Keep missing values as null. "
                    "Use Unknown for missing age groups."
                ),
                "statistics_policy": (
                    "Exclude missing values. "
                    "Use sample standard deviation (ddof=1)."
                ),
                "survival_rate_unit": "proportion from 0 to 1",
            },
            "summary": self.get_summary_stats(),
            "descriptive_statistics": self._table_to_records(
                self.descriptive_stats.rename_axis("column")
            ),
            "missing_values": self._table_to_records(
                self.missing_data.rename_axis("column")
            ),
            "family_size_by_survival": self._table_to_records(
                self.family_by_survival
            ),
            "feature_analysis": {
                name: self._table_to_records(table)
                for name, table in self.feature_analysis.items()
            },
            "passengers": [
                passenger.to_dict()
                for passenger in self.passengers
            ],
        }

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        print(f"\nJSON exported to: {filename}")
        print(f"Passenger records exported: {len(self.passengers)}")
        return data

# Create the dataset and check its summary.
dataset = TitanicDataset(
    dataframe=df_features,
    descriptive_stats=descriptive_stats,
    missing_data=missing_data,
    feature_analysis=feature_analysis,
    family_by_survival=family_by_survival,
)

print("\nDATASET CLASS CHECK")
for name, value in dataset.get_summary_stats().items():
    print(f"{name}: {value}")

# Export the complete dataset and analysis.
exported_data = dataset.to_json(JSON_FILE)

# Reject non-standard JSON constants such as NaN and Infinity.
def reject_invalid_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


# Load the exported file using strict JSON checks.
with open(JSON_FILE, "r", encoding="utf-8") as file:
    json_data = json.load(
        file,
        parse_constant=reject_invalid_constant,
    )

expected_keys = {
    "metadata",
    "summary",
    "descriptive_statistics",
    "missing_values",
    "family_size_by_survival",
    "feature_analysis",
    "passengers",
}

assert set(json_data) == expected_keys, "Unexpected JSON sections"

passengers = json_data["passengers"]
summary = json_data["summary"]

assert len(passengers) == len(df_features), "Passenger count mismatch"
assert json_data["metadata"]["total_passengers"] == len(passengers)
assert summary["total_passengers"] == len(passengers)

# Compare every passenger field with the source dataframe.
expected_passengers = [
    Passenger(row).to_dict()
    for _, row in df_features.iterrows()
]
assert passengers == expected_passengers, "Passenger data mismatch"

# Check summary counts against the original data.
assert summary["survived"] == int(df["Survived"].sum())
assert summary["did_not_survive"] == int((df["Survived"] == 0).sum())
assert (
    summary["survived"] + summary["did_not_survive"]
    == len(passengers)
)

# Check missing values against the original data.
missing_records = {
    item["column"]: item
    for item in json_data["missing_values"]
}
assert set(missing_records) == set(df.columns)

for column in df.columns:
    assert (
        missing_records[column]["missing_count"]
        == int(df[column].isna().sum())
    )

# Check that all numeric columns and engineered features are included.
statistics_columns = {
    item["column"]
    for item in json_data["descriptive_statistics"]
}
assert statistics_columns == set(numeric_data.columns)

assert set(json_data["feature_analysis"]) == {
    "FamilySize", "IsAlone", "AgeGroup"
}

for feature, records in json_data["feature_analysis"].items():
    assert sum(item["passengers"] for item in records) == len(df)
    assert sum(item["survived"] for item in records) == summary["survived"]

# Check that missing ages remain null after export.
assert sum(
    passenger["age"] is None
    for passenger in passengers
) == int(df["Age"].isna().sum())

print("\nJSON VALIDATION")
print("Strict JSON parsing: passed")
print(f"Passenger records checked: {len(passengers)}")
print("Passenger data and summary counts: passed")
print("Missing values and analysis structure: passed")
print("All validation checks passed.")