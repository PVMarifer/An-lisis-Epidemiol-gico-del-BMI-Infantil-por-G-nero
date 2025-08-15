# ====================================================
# Análisis Epidemiológico del BMI Infantil por Género
# ====================================================

# Este notebook analiza la distribución del BMI en niños y niñas
# a través de diferentes años escolares y hospitales.
# Incluye limpieza, transformación, normalización, análisis,
# visualización y exportación de resultados.

# ----------------------------
# Librerías
# ----------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

sns.set(style="whitegrid")

# Crear carpeta para exportar resultados
os.makedirs("results", exist_ok=True)

# ----------------------------
# Carga y exploración inicial
# ----------------------------
url = "BMIData.csv"
df = pd.read_csv(url)

print(f"Registros: {df.shape[0]}, Columnas: {df.shape[1]}\n")
display(df.head())
print("\nValores nulos por columna:")
print(df.isnull().sum())
print(f"\nDuplicados: {df.duplicated().sum()}\n")
display(df.describe())

# ----------------------------
# Limpieza y preparación de datos
# ----------------------------
df = df.drop_duplicates()
df.fillna(0, inplace=True)
df["ValidCounts"] = df["ValidCounts"].replace(0, pd.NA)

# ----------------------------
# Cálculo de porcentajes de BMI
# ----------------------------
bmi_cols = ["EpiUnderweight", "EpiHealthyWeight", "EpiOverweight", "EpiObese"]
for col in bmi_cols:
    df[col] = df[col].astype(float)

df["Pct_Underweight"] = df["EpiUnderweight"] / df["ValidCounts"]
df["Pct_HealthyWeight"] = df["EpiHealthyWeight"] / df["ValidCounts"]
df["Pct_Overweight"] = df["EpiOverweight"] / df["ValidCounts"]
df["Pct_Obese"] = df["EpiObese"] / df["ValidCounts"]

# ----------------------------
#Normalización de porcentajes
# ----------------------------
cols_pct = ["Pct_Underweight", "Pct_HealthyWeight", "Pct_Overweight", "Pct_Obese"]
scaler = MinMaxScaler()
df[cols_pct] = scaler.fit_transform(df[cols_pct])

# ----------------------------
# Resumen estadístico y análisis por categorías
# ----------------------------
# Obesidad promedio por año y género
obesidad_resumen = df.groupby(["SchoolYear","Sex"])["Pct_Obese"].mean().reset_index()
obesidad_resumen.to_csv("results/Obesidad_por_año_genero.csv", index=False)

# Hospitales con mayores tasas de sobrepeso
top_hosp = df.groupby("NameHospital")["Pct_Overweight"].mean().sort_values(ascending=False).head(15).reset_index()
top_hosp.to_csv("results/Top15_Hospitales_Overweight.csv", index=False)

# Exportar DataFrame final
df.to_csv("results/BMI_analysis_results.csv", index=False)

print("✅ Resultados exportados a la carpeta 'results/'")

# ----------------------------
# Matriz de correlaciones
# ----------------------------
corr = df[cols_pct + ["SchoolYear", "ValidCounts"]].corr()
display(corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlación entre tasas de BMI y variables")
plt.savefig("results/Correlaciones_BMI.png", bbox_inches='tight')
plt.close()

# ----------------------------
# Visualizaciones
# ----------------------------

# Pairplot por género
sns.pairplot(df, vars=cols_pct, hue="Sex")
plt.savefig("results/Pairplot_BMI.png", bbox_inches='tight')
plt.close()

# Evolución obesidad por año y género
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x="SchoolYear", y="Pct_Obese", hue="Sex", marker="o")
plt.title("Evolución del porcentaje de obesidad por año y género")
plt.ylabel("Porcentaje Obesidad Normalizado")
plt.savefig("results/Obesidad_por_año_genero.png", bbox_inches='tight')
plt.close()

# Boxplot por género
plt.figure(figsize=(8,5))
sns.boxplot(x="Sex", y="Pct_Obese", data=df)
plt.title("Distribución de obesidad por género")
plt.savefig("results/Distribución_Obesidad_por_Género.png", bbox_inches='tight')
plt.close()

# Histogramas por hospital
plt.figure(figsize=(12,6))
top_hosp_plot = df.groupby("NameHospital")["Pct_Obese"].mean().sort_values(ascending=False).head(15)
top_hosp_plot.plot(kind='bar', color='salmon')
plt.title("Hospitales con mayores tasas de obesidad promedio")
plt.ylabel("Porcentaje Obesidad Normalizado")
plt.xticks(rotation=45)
plt.savefig("results/Top15_Hospitales_Obesidad.png", bbox_inches='tight')
plt.close()

print("✅ Gráficos exportados a la carpeta 'results/'")
print("✅ Pipeline completo y listo para portafolio.")
