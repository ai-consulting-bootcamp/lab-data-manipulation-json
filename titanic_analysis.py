"""
Titanic Data Analysis and JSON Export
Author: [Angello Pagano]
Description: Analyze Titanic passenger data, engineer features, and export to JSON
"""
 
from os import name

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
 
# Set up paths
DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "titanic.csv"
JSON_FILE = DATA_DIR / "titanic_data.json"


 #tell pythin tomread the file using pndasd###


df= pd.read_csv(CSV_FILE)
print(df.head())
print(df.shape)
print(df.columns)
# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)
 
print("Project setup complete!")
print(f"Data directory: {DATA_DIR}")
print(f"CSV file location: {CSV_FILE}")


####descriptive analyses stufff ###


df['Age'].mean
print(df['Age'].mean())

df['Age'].median
print(df['Age'].median())

df['Age'].std
print(df['Age'].std())

df.select_dtypes
print(df.select_dtypes(include=[np.number]
                       ))
####define the variable numeric_columns to hold the numeric columns in the dataframe####
numeric_columns = df.select_dtypes(include=[np.number])
print("Numeric columns:", numeric_columns)

##calculate the meaan of the numeric columns######
##df[numeric_columns].mean()
print("Mean Values:")
print(numeric_columns.mean())

print("Median Values:")
print(numeric_columns.median())
print("Standard Deviation:")    
print(numeric_columns.std())


###checking for null vakes , which is  it the ame as missing values in the data frame####
df['Age'].isna()

print("Number of Null Values in 'Age' Column:")
print(df['Age'].isna().sum())
df.isna
print("Number of Null Values in Each Column:")
print(df.isna().sum())


###calculate percebtages###
print("Percentage of Null Values:")

missing_age_percentage = (df['Age'].isna().sum() / len(df)) * 100
print(f"Percentage of Null Values in 'Age' Column: {missing_age_percentage:.5f}%")


####calculate the percentage of missing values for all columns in the dataframe###
missing_percentages = (df.isna().sum() / len(df)) * 100
print("Percentage of Null Values in Each Column:")
print(missing_percentages)

##############################################################

###feature engineering and data cleaning####
print("Feature Engineering and Data Cleaning:")
# Create a copy of the dataframe for feature engineering
df_features = df.copy()
 
# Feature 1: Family Size
df_features['FamilySize'] = df_features['SibSp'] + df_features['Parch'] + 1
print(df_features[['SibSp', 'Parch', 'FamilySize']].head(10))


 
# Feature 2: Is Alone
df_features['IsAlone'] = df_features['FamilySize'] == 1
print(df_features[['FamilySize', 'IsAlone']].head(10))
 
# Feature 3: Age Groups
def categorize_age(age):
    """Categorize age into groups"""
    if pd.isna(age):
        return 'Unknown'
    elif age < 18:
        return 'Youngsters'
    elif age < 30:
        return 'Adults'
    elif age < 50:
        return 'Middle-aged'
    else:
        return 'Senior Citizens'
 
df_features['AgeGroup'] = df_features['Age'].apply(categorize_age)
print(df_features[['Age', 'AgeGroup']].head(10))
 
# Analyze feature differences between survivors and non-survivors
print("\n" + "="*50)
print("FEATURE ANALYSIS: SURVIVED vs NOT SURVIVED")
print("="*50)


 #####compute the mean family size for survivors and non-survivors#####
df_features.groupby('Survived')['FamilySize'].mean()

print("\nFamily Size by Survival:")
print((df_features.groupby('Survived')['FamilySize'].mean()))


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



##########what proportion of passengers survived in each group—alone versus not alone?###
is_alone_survival = df_features.groupby('IsAlone')['Survived'].mean()
print("\nProportion of Survivors by Aloneness:")
print(is_alone_survival)    

print('"False proportion means  not alone and survived and True proportion means alone and survived"')



#########lest comoare agefroup survival rates to see if age groups had different survival rates.####
age_group_survival = df_features.groupby('AgeGroup')['Survived'].mean()     
print("\nProportion of Survivors by Age Group:")
print(age_group_survival)   

df_features.groupby('AgeGroup').size
print("\nNumber of Passengers in Each Age Group:")
print(df_features.groupby('AgeGroup').size())   





###################pythin classes##########

# Step 3: Create Classes for JSON Export
 
class Passenger:
    """
    Represents a passenger with all their information.
    """
    def __init__(self, passenger_id, name, age, sex, survived, pclass, 
                 fare, embarked=None, family_size=None, is_alone=None, title=None, age_group=None):
   
   
            self.passenger_id = int(passenger_id) if pd.notna(passenger_id) else None
            self.name = name
            self.sex = sex
            self.survived = int(survived) if pd.notna(survived) else None
            self.pclass = int(pclass) if pd.notna(pclass) else None     
            self.fare = float(fare) if pd.notna(fare) else None
            self.embarked = embarked if pd.notna(embarked) else None
            self.family_size = int(family_size) if pd.notna(family_size) else None
            self.is_alone = bool(is_alone) if pd.notna(is_alone) else None
            self.title = title      
            self.age_group = age_group
            



            ####save the age paramter as self.age####
            self.age = (age) if pd.notna(age) else None    
   
   


##reating a passenger with made-up details and saving it in a variable. 
# Then try printing its ID ####


        # Tip: Use pd.notna() to check if a value is not null/NaN
        # Tip: Convert to appropriate types (int, float, str)
        # Example for passenger_id:

    
    def to_dict(self):
       ## """Convert passenger to dictionary for JSON serialization."""
##        # TODO: Return a dictionary with all passenger attributes
        # Tip: Dictionary keys should match the attribute names
        return {
            'passenger_id': self.passenger_id,
            'name': self.name,  
            'age': self.age,
            'sex': self.sex,
            'survived': self.survived,
            'pclass': self.pclass,
            'fare': self.fare,
            'embarked': self.embarked,
            'family_size': self.family_size,
            'is_alone': self.is_alone,
            'title': self.title,
            'age_group': self.age_group
        }   


passenger = Passenger(passenger_id=1, name="Angello Pagano", age=30, sex="male", survived=1, pclass=2, fare=20.0)
print(f"Passenger ID: {passenger.passenger_id}")
print(f"Passenger Name: {passenger.name}")
print(f"Passenger Age: {passenger.age}")
print(f"Passenger Sex: {passenger.sex}")
print(f"Passenger Survived: {passenger.survived}")
print(f"Passenger Pclass: {passenger.pclass}")
print(f"Passenger Fare: {passenger.fare}")
print(f"Passenger Embarked: {passenger.embarked}")
print(f"Passenger Family Size: {passenger.family_size}")
print(f"Passenger Is Alone: {passenger.is_alone}")
print(f"Passenger Title: {passenger.title}")

print(passenger.to_dict())  # Print the dictionary representation of the passenger  


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
            passenger_id = row['PassengerId'] if pd.notna(row['PassengerId']) else idx
            name = row['Name'] if pd.notna(row['Name']) else None   
            age = row['Age'] if pd.notna(row['Age']) else None
            sex = row['Sex'] if pd.notna(row['Sex']) else None
            survived = row['Survived'] if pd.notna(row['Survived']) else None
            pclass = row['Pclass'] if pd.notna(row['Pclass']) else None
            fare = row['Fare'] if pd.notna(row['Fare']) else None 
            embarked = row['Embarked'] if pd.notna(row['Embarked']) else None
            family_size = row['FamilySize'] if 'FamilySize' in row and pd.notna(row['FamilySize']) else None
            is_alone = row['IsAlone'] if 'IsAlone' in row and pd.notna(row['IsAlone']) else None
            title = row['Title'] if 'Title' in row and pd.notna(row['Title']) else None     




            # Example of getting PassengerId:
            # passenger_id = row.get('PassengerId', idx)
            
            passenger = Passenger(
            passenger_id=passenger_id,
            name=name,
            age=age,
            sex=sex,
            survived=survived,
            pclass=pclass,
            fare=fare,
            embarked=embarked,
            family_size=family_size,
            is_alone=is_alone,
            title=title,
            age_group=row['AgeGroup']
        )
            self.passengers.append(passenger)

            # TODO: Create the Passenger object and append to self.passengers
    
    def to_json(self, filename='titanic_data.json'):
        data = {
            'metadata': {
                'total_passengers': len(self.passengers),
                'dataset_name': 'Titanic Passenger Dataset',
                'survival_rate': float(self.dataframe['Survived'].mean()),
                'summary_statistics': self.get_summary_stats(),
                'missing_values': self.dataframe.isna().sum().to_dict(),
                'numeric_statistics': self.dataframe.select_dtypes(include=[np.number]).agg(['mean', 'median', 'std']).to_dict(),
                'exported_at': datetime.now().isoformat()
                
            },
            'passengers': [p.to_dict() for p in self.passengers]
        }

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)

        print(f"Data exported to {filename}")
        return data

    
    def get_summary_stats(self):
        """Get summary statistics."""
        # TODO: Calculate and return summary statistics
        # Tip: Use list comprehensions to filter and calculate
        # Example for counting survived passengers:
        # survived_count = sum(1 for p in self.passengers if p.survived == 1)
        return {
            'total_passengers': len(self.passengers),
            'survived': sum(1 for p in self.passengers if p.survived == 1),
            'did_not_survive': sum(1 for p in self.passengers if p.survived == 0),
            'average_age': float(self.dataframe['Age'].mean()) if not self.dataframe['Age'].isna().all() else None,
            'average_fare': float(self.dataframe['Fare'].mean()) if not self.dataframe['Fare'].isna().all() else None   
                    
        }
test_dataset = TitanicDataset(df_features.head(5))
full_dataset = TitanicDataset(df_features)
full_dataset.to_json(DATA_DIR / 'titanic_processed.json')
with open(DATA_DIR / 'titanic_processed.json', 'r', encoding='utf-8') as file:
    loaded_data = json.load(file)

print("Passengers in JSON:", len(loaded_data['passengers']))
print("Passengers in JSON:", len(loaded_data['passengers']))
print("JSON sections:", loaded_data.keys())
print(test_dataset.get_summary_stats())
test_dataset.to_json('test_passengers.json')
print("Summary Statistics:")
print(len(test_dataset.passengers))

print(test_dataset.passengers[0].to_dict())
 ## TODO: Create dataset object and export
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

