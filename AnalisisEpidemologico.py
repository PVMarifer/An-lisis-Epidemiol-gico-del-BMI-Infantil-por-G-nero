# ----------------------------
# 1️⃣ Librerías
# ----------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sns.set(style="whitegrid")  # Estilo profesional

# ----------------------------
# 2️⃣ Función de pipeline con Markdown simulado
# ----------------------------
def pipeline_BMI_pro(path):
    # 🔹 Carga de datos
    print("# ====================================================")
    print("# 1️⃣ Carga de datos")
    print("# ====================================================")
    df = pd.read_csv(path)
    print(f"Registros: {df.shape[0]}, Columnas: {df.shape[1]}\n")
    
    # 🔹 Exploración inicial
    print("## Exploración inicial")
    print("Primeras filas:")
    display(df.head())
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    print(f"Duplicados: {df.duplicated().sum()}\n")
    print(df.describe())
    
    # 🔹 Limpieza básica
    print("\n## Limpieza de datos")
    df = df.drop_duplicates()
    df.fillna(0, inplace=True)
    df["ValidCounts"] = df["ValidCounts"].replace(0, pd.NA)
    print("Limpieza completada ✅\n")
    
    # 🔹 Cálculo de porcentajes de BMI
    print("## Cálculo de porcentajes de BMI")
    bmi_cols = ["EpiUnderweight", "EpiHealthyWeight", "EpiOverweight", "EpiObese"]
    for col in bmi_cols:
        df[col] = df[col].astype(float)
    
    df["Pct_Underweight"] = df["EpiUnderweight"] / df["ValidCounts"]
    df["Pct_HealthyWeight"] = df["EpiHealthyWeight"] / df["ValidCounts"]
    df["Pct_Overweight"] = df["EpiOverweight"] / df["ValidCounts"]
    df["Pct_Obese"] = df["EpiObese"] / df["ValidCounts"]
    print("Porcentajes calculados ✅\n")
    
    # 🔹 Normalización
    print("## Normalización de porcentajes")
    cols_pct = ["Pct_Underweight", "Pct_HealthyWeight", "Pct_Overweight", "Pct_Obese"]
    scaler = MinMaxScaler()
    df[cols_pct] = scaler.fit_transform(df[cols_pct])
    print("Normalización completada ✅\n")
    
    # 🔹 Resumen estadístico
    print("## Resumen estadístico")
    print("📊 Obesidad promedio por año y género:")
    display(df.groupby(["SchoolYear","Sex"])["Pct_Obese"].mean())
    
    print("🏥 Hospitales con mayores tasas de sobrepeso:")
    display(df.groupby("NameHospital")["Pct_Overweight"].mean().sort_values(ascending=False).head(10))
    
    # 🔹 Correlaciones
    print("## Correlaciones entre variables")
    corr = df[cols_pct + ["SchoolYear", "ValidCounts"]].corr()
    display(corr)
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlación entre tasas de BMI y variables")
    plt.show()
    
    # 🔹 Visualizaciones
    print("## Visualizaciones")
    
    # Pairplot
    print("### Pairplot por género")
    sns.pairplot(df, vars=cols_pct, hue="Sex")
    plt.show()
    
    # Tendencia obesidad
    print("### Evolución del porcentaje de obesidad por año y género")
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x="SchoolYear", y="Pct_Obese", hue="Sex", marker="o")
    plt.title("Evolución del porcentaje de obesidad por año y género")
    plt.ylabel("Porcentaje Obesidad Normalizado")
    plt.show()
    
    # Boxplot por género
    print("### Distribución de obesidad por género")
    plt.figure(figsize=(8,5))
    sns.boxplot(x="Sex", y="Pct_Obese", data=df)
    plt.title("Distribución de obesidad por género")
    plt.show()
    
    # Histogramas por hospital
    print("### Hospitales con mayores tasas de obesidad promedio")
    top_hosp = df.groupby("NameHospital")["Pct_Obese"].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12,6))
    top_hosp.plot(kind='bar', color='salmon')
    plt.title("Hospitales con mayores tasas de obesidad promedio")
    plt.ylabel("Porcentaje Obesidad Normalizado")
    plt.xticks(rotation=45)
    plt.show()
    
    print("\n✅ Pipeline completado. Datos listos para análisis adicional o exportación.")
    
    return df

# ----------------------------
# 3️⃣ Ejecutar pipeline
# ----------------------------
url = "BMIData.csv"
df_final = pipeline_BMI_pro(url)
