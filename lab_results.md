# Titanic analysis: results and reflection

The analysis created FamilySize, IsAlone and AgeGroup to compare survival across family and age groups. In this dataset, 50.6% of passengers who were not alone survived, compared with 30.4% of those who were alone; the Youngsters group had a 54.0% survival rate. Classes helped organize each passenger's attributes and the dataset's export and summary methods.

## Selected program output

```text
Dataset shape: (891, 12)
Age mean: 29.699118; median: 28.0; standard deviation: 14.526497
Fare mean: 32.204208; median: 14.4542; standard deviation: 49.693429
Missing values: Age 177; Cabin 687; Embarked 2
Mean family size: survived 1.938596; did not survive 1.883424
Survival rate by IsAlone: False 0.505650; True 0.303538
Survival rate by AgeGroup: Youngsters 0.539823; Adults 0.350554;
  Middle-aged 0.417969; Senior Citizens 0.364865; Unknown 0.293785
Data exported to data/titanic_processed.json
Passengers in JSON: 891
```

Rates describe associations in this dataset and do not establish causation. Run `python titanic_analysis.py` from the directory containing the script and `data` folder; pandas and NumPy are required.

