"""Titanic data analysis, feature engineering, and JSON export."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "titanic.csv"
JSON_FILE = DATA_DIR / "titanic_processed.json"
DATA_DIR.mkdir(exist_ok=True)


def categorize_age(age):
    """Place an age into a readable age group."""
    if pd.isna(age):
        return "Unknown"
    if age < 18:
        return "Child"
    if age < 30:
        return "Young Adult"
    if age < 50:
        return "Adult"
    return "Senior"


class Passenger:
    """Represent one passenger using JSON-safe Python values."""

    def __init__(self, row):
        self.passenger_id = int(row["PassengerId"])
        self.name = str(row["Name"])
        self.age = float(row["Age"]) if pd.notna(row["Age"]) else None
        self.sex = str(row["Sex"])
        self.survived = int(row["Survived"])
        self.pclass = int(row["Pclass"])
        self.fare = float(row["Fare"]) if pd.notna(row["Fare"]) else None
        self.embarked = str(row["Embarked"]) if pd.notna(row["Embarked"]) else None
        self.family_size = int(row["FamilySize"])
        self.is_alone = bool(row["IsAlone"])
        self.age_group = str(row["AgeGroup"])

    def to_dict(self):
        return self.__dict__.copy()


class TitanicDataset:
    """Store processed passengers and export them with metadata."""

    def __init__(self, dataframe, statistics, missing_data):
        self.dataframe = dataframe
        self.statistics = statistics
        self.missing_data = missing_data
        self.passengers = [Passenger(row) for _, row in dataframe.iterrows()]

    def get_summary_stats(self):
        ages = [p.age for p in self.passengers if p.age is not None]
        fares = [p.fare for p in self.passengers if p.fare is not None]
        survived = sum(p.survived == 1 for p in self.passengers)
        return {
            "total_passengers": len(self.passengers),
            "survived": survived,
            "did_not_survive": len(self.passengers) - survived,
            "survival_rate": survived / len(self.passengers),
            "average_age": sum(ages) / len(ages),
            "average_fare": sum(fares) / len(fares),
        }

    def to_json(self, filename):
        summary = self.get_summary_stats()
        data = {
            "metadata": {
                "dataset_name": "Titanic Passenger Dataset",
                "export_date": datetime.now().isoformat(),
                "summary_statistics": summary,
                "engineered_features": ["FamilySize", "IsAlone", "AgeGroup"],
                "missing_values": {
                    column: {
                        "missing_count": int(row["missing_count"]),
                        "missing_percentage": float(row["missing_percentage"]),
                    }
                    for column, row in self.missing_data.iterrows()
                },
            },
            "passengers": [passenger.to_dict() for passenger in self.passengers],
        }

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        return data


def main():
    # Load the source data and inspect its shape and first rows.
    df = pd.read_csv(CSV_FILE)
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    print(df.head())

    # Calculate descriptive statistics for every numeric column.
    numeric_columns = df.select_dtypes(include=np.number).columns
    statistics = df[numeric_columns].agg(["mean", "median", "std"])
    print("\nDESCRIPTIVE STATISTICS")
    print(statistics)

    # Count missing values and calculate their percentages.
    missing_count = df.isna().sum()
    missing_data = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": (missing_count / len(df) * 100).round(2),
    })
    print("\nMISSING VALUES ANALYSIS")
    print(missing_data)

    # Create features that may help explain survival patterns.
    df_features = df.copy()
    df_features["FamilySize"] = df_features["SibSp"] + df_features["Parch"] + 1
    df_features["IsAlone"] = df_features["FamilySize"] == 1
    df_features["AgeGroup"] = df_features["Age"].apply(categorize_age)

    print("\nFEATURE ANALYSIS")
    print(df_features.groupby("Survived")["FamilySize"].agg(["mean", "median", "std"]))
    print(df_features.groupby("Survived")["IsAlone"].mean())
    print(pd.crosstab(df_features["AgeGroup"], df_features["Survived"]))

    # Convert the processed rows into objects and export them as JSON.
    dataset = TitanicDataset(df_features, statistics, missing_data)
    print("\nSUMMARY STATISTICS")
    print(dataset.get_summary_stats())
    dataset.to_json(JSON_FILE)
    print(f"Data exported to {JSON_FILE}")

    # Reload the file to validate its structure and record count.
    with open(JSON_FILE, encoding="utf-8") as file:
        json_data = json.load(file)
    assert set(json_data) == {"metadata", "passengers"}
    assert len(json_data["passengers"]) == len(df_features)
    print(f"JSON validation passed! Passenger records: {len(json_data['passengers'])}")


if __name__ == "__main__":
    main()
