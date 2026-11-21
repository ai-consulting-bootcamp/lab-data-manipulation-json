import json
from datetime import datetime
from pathlib import Path

import pandas as pd

carpeta_proyecto = Path(__file__).parent
DATA_DIR = carpeta_proyecto / "data"
DATA_DIR.mkdir(exist_ok=True)

archivo_csv = DATA_DIR / "titanic.csv"
if not archivo_csv.exists():
    raise FileNotFoundError(f"No se encontró el archivo CSV: {archivo_csv}")

df = pd.read_csv(archivo_csv)

print("Datos cargados correctamente")
print("Filas y columnas:", df.shape)
print("Columnas:", list(df.columns))
print("Primeras filas:")
print(df.head())

# Calcular estadísticas de las columnas numéricas
columnas_numericas = df.select_dtypes(include="number").columns

print("\nEstadísticas descriptivas:")
for columna in columnas_numericas:
    print(f"\n{columna}")
    print(f"Media: {df[columna].mean():.2f}")
    print(f"Mediana: {df[columna].median():.2f}")
    print(f"Desviación estándar: {df[columna].std():.2f}")
print("\nEstadísticas avanzadas:")

print("\nCuartiles de las columnas numéricas:")
print(df[columnas_numericas].quantile([0.25, 0.50, 0.75]))

print("\nAsimetría (skewness):")
print(df[columnas_numericas].skew())

print("\nCurtosis (kurtosis):")
print(df[columnas_numericas].kurt())

matriz_correlacion = df[columnas_numericas].corr()

print("\nMatriz de correlación:")
print(matriz_correlacion)

print("\nCorrelación con supervivencia:")
print(matriz_correlacion["Survived"].sort_values(ascending=False))
print("\nValores faltantes por columna:")
print(df.isna().sum())

# Crear una variable con el tamaño de la familia a bordo
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print("\nTamaño de familia (primeras filas):")
print(df[["SibSp", "Parch", "FamilySize"]].head())

# Indicar si el pasajero viajaba solo
df["IsAlone"] = df["FamilySize"] == 1
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
print("\n¿Viajaba solo? (primeras filas):")
print(df[["FamilySize", "IsAlone"]].head())

print("\nTamaño promedio de familia según supervivencia:")
print(df.groupby("Survived")["FamilySize"].mean())

print("\nProporción que viajaba sola según supervivencia:")
print(df.groupby("Survived")["IsAlone"].mean())

class Passenger:
    def __init__(self, passenger_id, name, age, sex, survived, pclass, fare, embarked=None, family_size=None, is_alone=None, title=None):

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
     self.title = str(title) if pd.notna(title) else None



    def to_dict(self):
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
    
class TitanicDataset:
    """Representa el conjunto de datos del Titanic."""

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.passengers = []
        self._create_passengers()

    def _create_passengers(self):
        """Crea un objeto Passenger por cada fila del DataFrame."""
        for idx, row in self.dataframe.iterrows():
            passenger = Passenger(
                passenger_id=row.get("PassengerId", idx),
                name=row.get("Name"),
                age=row.get("Age"),
                sex=row.get("Sex"),
                survived=row.get("Survived"),
                pclass=row.get("Pclass"),
                fare=row.get("Fare"),
                embarked=row.get("Embarked"),
                family_size=row.get("FamilySize"),
                is_alone=row.get("IsAlone"),
                title=row.get("Title"),
            )
            self.passengers.append(passenger)

    def to_json(self, filename="titanic_data.json"):
        """Exporta los datos del Titanic a un archivo JSON."""
        data = {
            "metadata": {
                "dataset_name": "Titanic Passenger Dataset",
                "export_date": datetime.now().isoformat(),
                "total_passengers": len(self.passengers),
                "survival_rate": float(self.dataframe["Survived"].mean()),
                "summary_stats": self.get_summary_stats(),
                "missing_values": {
                   column: int(count)
                   for column, count in self.dataframe.isna().sum().items()
            },
        },
            "passengers": [passenger.to_dict() for passenger in self.passengers],
        }

        with open(filename, "w", encoding="utf-8") as archivo:
            json.dump(data, archivo, indent=2, ensure_ascii=False)

        print(f"Datos exportados a {filename}")
        return data

    def get_summary_stats(self):
        """Calcula estadísticas resumidas del conjunto de datos."""
        total_passengers = len(self.passengers)
        survived_count = sum(
            1 for passenger in self.passengers if passenger.survived == 1
        )
        did_not_survive_count = sum(
            1 for passenger in self.passengers if passenger.survived == 0
        )

        ages = [passenger.age for passenger in self.passengers if passenger.age is not None]
        fares = [passenger.fare for passenger in self.passengers if passenger.fare is not None]

        return {
            "total_passengers": total_passengers,
            "survived": survived_count,
            "did_not_survive": did_not_survive_count,
            "average_age": sum(ages) / len(ages) if ages else None,
            "average_fare": sum(fares) / len(fares) if fares else None,
        }

# Crear el conjunto de datos y mostrar sus estadísticas
dataset = TitanicDataset(df)

print("\nEstadísticas resumidas:")
for nombre, valor in dataset.get_summary_stats().items():
    print(f"{nombre}: {valor}")

archivo_json = carpeta_proyecto / "data" / "titanic_processed.json"
dataset.to_json(archivo_json)

with open(archivo_json, "r", encoding="utf-8") as archivo:
    json_data = json.load(archivo)

print("\nValidación del JSON:")
print("Secciones principales:", list(json_data.keys()))
print("Claves de metadata:", list(json_data["metadata"].keys()))
print("Pasajeros exportados:", len(json_data["passengers"]))
print("Coinciden con DataFrame:", len(json_data["passengers"]) == len(df))
print("Incluye estadísticas:", "summary_stats" in json_data["metadata"])
print("Incluye valores faltantes:", "missing_values" in json_data["metadata"])