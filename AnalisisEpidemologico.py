import pandas as pd

# Carga del dataset
url = "BMIData.csv"
df = pd.read_csv(url)

# Revisar registros, tipos y nulos
print(f"Registros: {df.shape[0]}")
print(df.dtypes)
print("Valores nulos por columna:")
print(df.isnull().sum())

# Buscar duplicados
print(f"Duplicados: {df.duplicated().sum()}")

# Estadísticas básicas para detectar outliers o inconsistencias
print(df.describe())

# Opcional: ver primeras filas
print(df.head())

df["Pct_Underweight"] = df["EpiUnderweight"] / df["ValidCounts"]
df["Pct_HealthyWeight"] = df["EpiHealthyWeight"] / df["ValidCounts"]
df["Pct_Overweight"] = df["EpiOverweight"] / df["ValidCounts"]
df["Pct_Obese"] = df["EpiObese"] / df["ValidCounts"]

# Resumen tasas de obesidad por año y género
print(df.groupby(["SchoolYear", "Sex"])["Pct_Obese"].mean())

# Hospitales con mayores índices de sobrepeso
print(df.groupby("NameHospital")["Pct_Overweight"].mean().sort_values(ascending=False).head(10))

from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Normalizar porcentajes para comparar en rango 0-1
cols_pct = ["Pct_Underweight", "Pct_HealthyWeight", "Pct_Overweight", "Pct_Obese"]
scaler = MinMaxScaler()
df[cols_pct] = scaler.fit_transform(df[cols_pct])

# Matriz de correlaciones
corr = df[cols_pct + ["SchoolYear", "ValidCounts"]].corr()
print(corr)

# Heatmap para visualizar correlaciones
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlación entre tasas BMI y variables")
plt.show()

# Gráficos de dispersión para diferencias por género
sns.pairplot(df, vars=cols_pct, hue="Sex")
plt.show()

# Tendencia obesidad por año
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x="SchoolYear", y="Pct_Obese", hue="Sex", marker="o")
plt.title("Evolución del porcentaje de obesidad por año y género")
plt.ylabel("Porcentaje Obesidad Normalizado")
plt.show()

# Boxplot comparativo por género
plt.figure(figsize=(8,5))
sns.boxplot(x="Sex", y="Pct_Obese", data=df)
plt.title("Distribución de obesidad por género")
plt.show()

# Histograma de BMI por hospital (por ejemplo obesidad)
plt.figure(figsize=(12,6))
df.groupby("NameHospital")["Pct_Obese"].mean().sort_values(ascending=False).head(15).plot(kind='bar')
plt.title("Hospitales con mayores tasas de obesidad promedio")
plt.ylabel("Porcentaje Obesidad Normalizado")
plt.xticks(rotation=45)
plt.show()
