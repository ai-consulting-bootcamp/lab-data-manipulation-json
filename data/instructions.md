# Data Manipulation + JSON Handling

> **How you'll submit this lab**
>
> This repo is your lab. Fork it, do the work described below in your fork, then open a pull
> request back into this repository. An AI reviewer will check your PR against `rubric.md` and
> leave feedback directly on the PR. See `README.md` for the full workflow.

**Scenario**  
You're working as a data analyst for a maritime safety research organization. Your team has been given access to the famous Titanic passenger dataset to analyze survival patterns. Your task is to clean the data, perform statistical analysis, engineer new features that might predict survival, and export the processed data to JSON format for use in other systems.

**Learning Objectives**
- [ ] Download and import CSV data from a public source
- [ ] Calculate descriptive statistics (mean, median, standard deviation) for numeric columns
- [ ] Identify and count missing values in the dataset
- [ ] Perform feature engineering by creating new variables from existing ones
- [ ] Analyze how engineered features differentiate between groups (survived vs. not survived)
- [ ] Export data to JSON format using Python classes
- [ ] Validate JSON output structure and content

**Estimated Time:** 120-150 minutes

**Prerequisites:**
- [ ] Basic understanding of Python syntax
- [ ] Familiarity with pandas library
- [ ] Knowledge of dictionaries and data structures
- [ ] Understanding of basic statistics (mean, median, standard deviation)

---

## Introduction

Welcome to the Data Manipulation and JSON Handling lab! In this exercise, you'll work with real-world data from the Titanic disaster to practice essential data analysis skills. This lab will help you understand how to:

**What you'll build:**
- A data analysis script that processes the Titanic dataset
- Feature engineering functions that create new predictive variables
- A class-based system to structure and export data to JSON
- Statistical analysis comparing survival groups

**Why this matters:**
Data manipulation and feature engineering are fundamental skills in AI and machine learning. Most real-world datasets require cleaning, transformation, and the creation of new features before they can be used effectively. JSON is a universal data format used in APIs, web applications, and data pipelines.

**Success criteria:**
- [ ] Successfully download and import the Titanic dataset
- [ ] Calculate all required statistics correctly
- [ ] Identify missing values accurately
- [ ] Create at least 2 engineered features
- [ ] Export data to JSON using classes
- [ ] Validate JSON structure and content

---

## Background Story

The RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in 1912 after striking an iceberg. The disaster resulted in the deaths of over 1,500 passengers and crew. The dataset contains information about 891 passengers, including whether they survived or not.

Your research team wants to understand:
- What statistical patterns exist in the data?
- Can we create new features that better predict survival?
- How can we structure this data for use in other systems?

---

## Step-by-Step Instructions

### Step 1: Setting Up the Project

**Objective:** Create your project structure and download the dataset.

**What to do:**
1. Create a new Python file named `titanic_analysis.ipynb` or `titanic_analysis.py`
2. Create a folder named `data` in the same directory as your script
3. Download the Titanic dataset from Kaggle

**Dataset Download:**
The Titanic dataset is available on Kaggle. You can download it from:
- **Direct Link:** [Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic/data)
- **Alternative:** If you don't have Kaggle access, you can use this direct download link: [Titanic train.csv](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

**Instructions:**
1. Download the `train.csv` file
2. Save it in your `data` folder as `titanic.csv`
3. Make sure your file structure looks like this:
```
your-project/
├── titanic_analysis.py
└── data/
    └── titanic.csv
```

**Code template:**
```python
"""
Titanic Data Analysis and JSON Export
Author: [Your Name]
Description: Analyze Titanic passenger data, engineer features, and export to JSON
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
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
```

Tip: if you are not using the NumPy library, this code will need to be adapted.

**Expected outcome:** You should have a project structure with the data folder and your Python script ready.

**Checkpoint:** Run your script to verify the setup works. It should print the directory paths without errors.

---

### Step 2: Importing and Exploring the Data

**Objective:** Load the CSV file into a pandas DataFrame and get an initial understanding of the data.

**What to do:**
1. Use pandas to read the CSV file
2. Display basic information about the dataset
3. View the first few rows

**Code template for verification:**
```python
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
```
---

### Step 3: Calculating Descriptive Statistics

**Objective:** Calculate mean, median, and standard deviation for numeric columns.

**What to do:**
1. Identify all numeric columns in the dataset
2. Calculate mean, median, and standard deviation for each numeric column
3. Display the results in a clear format

**Code template:**
```


# Select numeric columns only
numeric_columns = 'your code here'
# Calculate statistics (.mean, median, std)

```

**Expected outcome:** You should see mean, median, and standard deviation calculated for columns like Age, Fare, SibSp, Parch, etc.

**Checkpoint:** Verify that your statistics make sense. For example, Age should have a mean around 29-30, and Fare should have a mean around 32.

---

### Step 4: Identifying Missing Values

**Objective:** Count and analyze missing values in the dataset.

**What to do:**
1. Count missing values for each column
2. Calculate the percentage of missing values
3. Identify which columns have the most missing data

**Code template:**
```python
# Count missing values
print("\n" + "="*50)
print("MISSING VALUES ANALYSIS")
print("="*50)

missing_data = {}

for col in df.columns:
    missing_count = 'your code here'
    missing_percent = (missing_count / len(df)) * 100
    
```

**Expected outcome:** You should identify that Age has missing values (around 177 missing, around 20%), Cabin has many missing values (around 77%), and Embarked has a few missing values (around 0.2%).

**Checkpoint:** Verify that your missing value counts match the expected values. Age should have approximately 177 missing values.

---

### Step 5: Feature Engineering

**Objective:** Create new features that might help differentiate between survivors and non-survivors.

**What to do:**
1. Create a "FamilySize" feature (SibSp + Parch + 1)
2. Create an "IsAlone" feature (FamilySize == 1)
3. Create an "AgeGroup" feature (categorize age into groups)
4. Analyze how these features differ between survivors and non-survivors

**Code template:**
```python
# Create a copy of the dataframe for feature engineering
df_features = df.copy()

# Feature 1: Family Size
df_features['FamilySize'] = 'your code here'
print(df_features[['SibSp', 'Parch', 'FamilySize']].head(10))

# Feature 2: Is Alone
df_features['IsAlone'] = 'your code here'
print(df_features[['FamilySize', 'IsAlone']].head(10))

# Feature 3: Age Groups
def categorize_age(age):
    """Categorize age into groups"""
    if pd.isna(age):
        return 'Unknown'
    elif age < 18:
        return 'name your category'
    elif age < 30:
        return 'name your category'
    elif age < 50:
        return 'name your category'
    else:
        return 'name your category'

df_features['AgeGroup'] = 'your code here'
print(df_features[['Age', 'AgeGroup']].head(10))

# Analyze feature differences between survivors and non-survivors
print("\n" + "="*50)
print("FEATURE ANALYSIS: SURVIVED vs NOT SURVIVED")
print("="*50)

# Here is an example to get you started:
print("\nFamily Size by Survival:")
family_survival = df_features.groupby('Survived')['FamilySize'].agg(['mean', 'median', 'std'])
print(family_survival)

# Statistical test: Do these features help differentiate?
print("\n" + "="*50)
print("FEATURE DIFFERENTIATION ANALYSIS")
print("="*50)

survived = df_features[df_features['Survived'] == 1]
not_survived = df_features[df_features['Survived'] == 0]

print("\nFamily Size:")
print(f"  Survived mean: {survived['FamilySize'].mean():.2f}")
print(f"  Not Survived mean: {not_survived['FamilySize'].mean():.2f}")
print(f"  Difference: {abs(survived['FamilySize'].mean() - not_survived['FamilySize'].mean()):.2f}")
```

**Checkpoint:** Verify that your engineered features are created correctly and show meaningful differences between survival groups. Can you think about any other new feature? What insights do these features provide?

---

### Step 6: Creating a Data Export Class

**Objective:** Create a Python class to structure and export data to JSON format.

**What to do:**
1. Create a class that encapsulates the data processing logic. We will structure two different classes: one for Passenger and  another TitanicDataset
2. Include the information we used to calculate statistics, identify missing values, and engineering features that you have previously done.
3. Add a method to export everything to JSON

Tip: It is good practice to encapsulate related functionality within a class and to take your time to structure it well. It will make your code cleaner and easier to maintain on the long run. If you want to take this one step further, you can also add new methods inside TitanicDataset for handling statistics, missing values, and feature engineering.

**Code template:**

```python
# Step 3: Create Classes for JSON Export

class Passenger:
    """
    Represents a passenger with all their information.
    """
    def __init__(self, passenger_id, name, age, sex, survived, pclass, 
                 fare, embarked=None, family_size=None, is_alone=None, title=None):
        # TODO: Initialize all passenger attributes
        # Tip: Use pd.notna() to check if a value is not null/NaN
        # Tip: Convert to appropriate types (int, float, str)
        # Example for passenger_id:
        self.passenger_id = int(passenger_id) if pd.notna(passenger_id) else None
        
        # TODO: Complete initialization for remaining attributes
        # Remember to handle None/NaN values appropriately
        pass
    
    def to_dict(self):
        """Convert passenger to dictionary for JSON serialization."""
        # TODO: Return a dictionary with all passenger attributes
        # Tip: Dictionary keys should match the attribute names
        return {
            'passenger_id': self.passenger_id,
            # TODO: Add all other attributes here
        }

class TitanicDataset:
    """
    Represents the entire Titanic dataset with methods for JSON export.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.passengers = []  # Will store Passenger objects
        self._create_passengers()
    
    def _create_passengers(self):
        """Create Passenger objects from dataframe."""
        # TODO: Iterate through dataframe rows and create Passenger objects
        # Tip: Use self.dataframe.iterrows() to loop through rows
        # Tip: Use row.get('ColumnName', default_value) to safely get values
        for idx, row in self.dataframe.iterrows():
            # TODO: Create a Passenger object with data from the row
            # Example of getting PassengerId:
            # passenger_id = row.get('PassengerId', idx)
            
            # TODO: Create the Passenger object and append to self.passengers
            pass
    
    def to_json(self, filename='titanic_data.json'):
        """Export dataset to JSON file."""
        # TODO: Create a dictionary with metadata and passenger data
        data = {
            'metadata': {
                'dataset_name': 'Titanic Passenger Dataset',
                'export_date': datetime.now().isoformat(),
                # TODO: Add more metadata fields
                # Tip: Calculate total_passengers from self.passengers
                # Tip: Calculate survival_rate using self.dataframe['Survived'].mean()
            },
            'passengers': []  # TODO: Convert all passenger objects to dictionaries
            # Tip: Use list comprehension with p.to_dict() for each passenger
        }
        
        # TODO: Write the data to a JSON file
        # Tip: Use json.dump() with indent=2 for readable formatting
        
        print(f"Data exported to {filename}")
        return data
    
    def get_summary_stats(self):
        """Get summary statistics."""
        # TODO: Calculate and return summary statistics
        # Tip: Use list comprehensions to filter and calculate
        # Example for counting survived passengers:
        # survived_count = sum(1 for p in self.passengers if p.survived == 1)
        
        return {
            'total_passengers': None,  # TODO: Calculate total passengers
            'survived': None,  # TODO: Count passengers who survived
            'did_not_survive': None,  # TODO: Count passengers who didn't survive
            # TODO: Add more statistics (average_age, average_fare)
            # Tip: Remember to handle None values when calculating averages
        }

# TODO: Create dataset object and export
# Check if df_engineered exists and is not empty
if 'df_engineered' in locals() and not df_engineered.empty:
    # TODO: Create a TitanicDataset object
    # dataset = TitanicDataset(...)
    
    # TODO: Print basic information about the dataset
    
    # TODO: Get and display summary statistics
    # Tip: Call get_summary_stats() and iterate through the results
    
    # TODO: Export to JSON (optional - uncomment when ready)
    # dataset.to_json('titanic_data.json')
    
    pass  # Remove this when you add your code
```

**Expected outcome:** You should have a complete class that processes the data and exports it to JSON format.

**Checkpoint:** Run your script and verify that:
1. The class loads data correctly
2. Statistics are calculated
3. Missing values are analyzed
4. Features are engineered
5. JSON file is created successfully
6. JSON validation passes

---

### Step 7: Testing and Validation

**Objective:** Verify that your JSON export is correct and complete.

**What to do:**
1. Load the JSON file and inspect its structure
2. Verify that all data is present
3. Check that the JSON is valid and can be parsed

**Code template:**
```python
# Additional validation: Load and inspect JSON
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# Print summary of JSON data and verify content
```

**Expected outcome:** You should see the JSON file loaded successfully with all expected data structures.

**Checkpoint:** Verify that your JSON file:
- Can be loaded without errors
- Contains all expected keys
- Has the correct number of passenger records
- Includes all statistics and missing value information

---

## Submission Guidelines

### What to Submit

**Required deliverables:**
- [ ] Your complete `titanic_analysis.py` file with all code implemented
- [ ] The exported `titanic_processed.json` file from the `data` folder
- [ ] A screenshot or text output showing:
  - Statistics calculations
  - Missing value analysis
  - Feature engineering results
  - JSON export confirmation
- [ ] A brief explanation (2-3 sentences) describing:
  - Which engineered features you created
  - How these features help differentiate between survivors and non-survivors
  - What you learned about using classes for data processing

### How to Submit

**Upload your work:**
1. **For code files:** Upload your `titanic_analysis.py` file
2. **For JSON file:** Upload your `titanic_processed.json` file
3. **For screenshots:** Upload PNG or JPG images showing your output
4. **For text output:** Copy and paste the output into the submission field

**Important notes:**
- Make sure your code runs without errors
- Include comments in your code explaining key sections
- Test your JSON file by trying to load it in a JSON validator
- Your JSON should be properly formatted and readable

**Due date:** Check with your instructor for the specific due date.

---

## Troubleshooting

**Common issues and solutions:**

**Issue 1: "FileNotFoundError: data/titanic.csv"**
- **Solution:** Make sure you've downloaded the CSV file and placed it in the `data` folder. Check that the file path is correct and the file exists.

**Issue 2: "ModuleNotFoundError: No module named 'pandas'"**
- **Solution:** Install pandas using `pip install pandas` or `pip install pandas numpy` if you also need numpy.

**Issue 3: "JSON serialization error"**
- **Solution:** Make sure you're converting numpy types to native Python types (use `float()`, `int()`, etc.). The `json.dump()` function cannot serialize numpy types directly.

**Issue 4: "Statistics showing NaN values"**
- **Solution:** This is normal for columns with missing values. You can use `df[col].mean(skipna=True)` to handle NaN values, or fill them first.

**Issue 5: "Feature engineering not showing differences"**
- **Solution:** Make sure you're comparing the right groups. Use `df[df['Survived'] == 1]` for survivors and `df[df['Survived'] == 0]` for non-survivors.

**Issue 6: "JSON file too large or unreadable"**
- **Solution:** Consider exporting only a subset of columns or passengers if the file is too large. You can use `df.head(100).to_dict('records')` to export just the first 100 rows for testing.

---

## Bonus Challenges

If you finish early, try these additional challenges:

### Challenge 1: Additional Feature Engineering
Create more sophisticated features:
- **Title Extraction:** Extract titles (Mr., Mrs., Miss, etc.) from the Name column
- **Fare per Person:** Calculate fare divided by family size
- **Cabin Deck:** Extract deck letter from Cabin (A, B, C, etc.) if available
- Analyze which of these new features best differentiates survivors

### Challenge 2: Advanced Statistics
Calculate additional statistics:
- **Quartiles:** Calculate Q1, Q2 (median), Q3 for numeric columns
- **Skewness and Kurtosis:** Measure data distribution shape
- **Correlation Matrix:** Find correlations between numeric features and survival

### Challenge 3: Data Visualization
Create visualizations to explore your features:
- Bar charts comparing survival rates by age group
- Histograms showing family size distribution for survivors vs. non-survivors
- Box plots comparing fare distributions

### Challenge 4: JSON Schema Validation
Create a JSON schema to validate your exported data structure:
- Define expected structure using JSON Schema format
- Validate your exported JSON against the schema
- Add schema validation to your class

---

## Additional Resources

**Reference materials:**
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Python JSON Module](https://docs.python.org/3/library/json.html)
- [Python Classes Tutorial](https://docs.python.org/3/tutorial/classes.html)
- [Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic/data)

**Tools and software:**
- [JSON Validator](https://jsonlint.com/) - Validate your JSON files
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [VS Code](https://code.visualstudio.com/) - Recommended code editor

**Further reading:**
- [Feature Engineering Best Practices](https://www.kaggle.com/learn/feature-engineering)
- [Data Cleaning with Pandas](https://realpython.com/python-data-cleaning-numpy-pandas/)
- [Working with JSON in Python](https://realpython.com/python-json/)

---

## To-Do List

- [ ] Read and understand the lab requirements
- [ ] Complete Step 1: Setting Up the Project
- [ ] Complete Step 2: Importing and Exploring the Data
- [ ] Complete Step 3: Calculating Descriptive Statistics
- [ ] Complete Step 4: Identifying Missing Values
- [ ] Complete Step 5: Feature Engineering
- [ ] Complete Step 6: Creating a Data Export Class
- [ ] Complete Step 7: Testing and Validation
- [ ] Test your code with different scenarios
- [ ] Review your code and add comments
- [ ] Submit your work

---

## Learning Reflection

After completing this lab, reflect on what you've learned:

1. **Data Import:** How did you handle downloading and importing external data?
2. **Statistical Analysis:** What insights did you gain from calculating mean, median, and standard deviation?
3. **Missing Values:** Why is it important to identify and understand missing data?
4. **Feature Engineering:** How did creating new features help you understand the data better?
5. **Classes:** What advantages did using a class provide over writing functions separately?
6. **JSON Export:** Why is JSON a useful format for data exchange?

**Key takeaways:**
- Data manipulation is a crucial first step in any data science project
- Feature engineering can reveal patterns that aren't obvious in raw data
- Classes help organize code and make it reusable
- JSON is a universal format for data exchange between systems
- Understanding your data through statistics and exploration is essential before building models

These skills form the foundation for more advanced machine learning and AI work.

---

*Good luck with your Titanic data analysis! Remember to explore the data, ask questions, and don't hesitate to experiment with different features.*
