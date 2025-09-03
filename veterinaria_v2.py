import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# Sirve para ocultar los mensajes de warning para versiones futuras
import warnings; warnings.filterwarnings('ignore')


data_excel = pd.read_csv('content\data_patitas_felices_v2.csv')

df = pd.DataFrame(data_excel)

# 2. Exploración y Limpieza Inicial de Datos
print("Primeras filas del dataset:\n", df.head())
print("\nInformación del dataset:\n", df.info())
print("\nDescripción del dataset:\n", df.describe())

# a. Manejo de datos faltantes (Imputación simple con la moda para datos categóricos y la mediana para datos numéricos)
for column in df.columns:
    if df[column].isnull().any():
        if df[column].dtype == 'object':  # Si es categórica
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:  # Si es numérica
            df[column].fillna(df[column].median(), inplace=True)
print("\nDatos faltantes después de la imputación:\n", df.isnull().sum()) #Verificamos que no tengamos nulos

df.head()

# b. Conversión de variables categóricas a numéricas
#Label Encoding
label_encoder = LabelEncoder()
df['tamaño'] = label_encoder.fit_transform(df['tamaño'])
df.head()

#Dummy Encoding
df = pd.get_dummies(df, columns=['especie', 'genero', 'raza'], drop_first=True)
df.head()

df.info()

# c. Escalado de características numéricas
scaler = StandardScaler()
df[['edad','tamaño','tiempo_en_adopcion']] = scaler.fit_transform(df[['edad','tamaño', 'tiempo_en_adopcion']])
df.head()

#Visualizamos un diagrama de correlacion para ver el comportamiento de las variables
plt.figure(figsize=(25,25))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# 3. Preparación de los datos para el modelo
# a. Definir la variable objetivo y las características
X = df.drop('adoptado_en_30_dias', axis=1)  # Características
y = df['adoptado_en_30_dias']  # Variable objetivo

# b. Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 4. Inicializar, entrenar y predecir con el modelo
# a. Inicializar el modelo (Regresión Logística)
model = LogisticRegression(random_state=42, solver='liblinear')

# b. Entrenar el modelo
model.fit(X_train, y_train)

# c. Predecir con el conjunto de prueba
y_pred = model.predict(X_test)

# 5. Evaluación del Modelo
# a. Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2%}')

# b. Imprimir el reporte de clasificación
print('\nReporte de Clasificación:\n', classification_report(y_test, y_pred))


# Datos de la nueva mascota
nueva_mascota = {
    'edad': 24,
    'tamaño': 'pequeño',
    'tiempo_en_adopcion': 60,
    'especie': 'conejo',
    'genero': 'macho',
    'raza': 'Persian'
}

# Crear un DataFrame con los datos de la nueva mascota
nueva_mascota_df = pd.DataFrame([nueva_mascota])

# Preprocesamiento
# 1. Label Encoding para 'tamaño'
nueva_mascota_df['tamaño'] = label_encoder.transform(nueva_mascota_df['tamaño'])

# 2. Dummy Encoding para 'especie', 'genero' y 'raza'
nueva_mascota_df = pd.get_dummies(nueva_mascota_df,
                                  columns=['especie', 'genero', 'raza'],
                                  drop_first=True)

# Asegurarse de que todas las columnas necesarias existen
# (esto es crucial si la nueva mascota tiene una raza/especie/genero no vista en el entrenamiento)
for col in X_train.columns:
    if col not in nueva_mascota_df.columns:
        nueva_mascota_df[col] = 0  # Añadir la columna con valor 0

# 3. Escalado de características numéricas
columnas_a_escalar = ['edad', 'tamaño', 'tiempo_en_adopcion']

#Asegurarse que solo existen columnas a escalar en el dataframe
columnas_existentes = [col for col in columnas_a_escalar
                       if col in nueva_mascota_df.columns]
nueva_mascota_df[columnas_existentes] = scaler.transform(nueva_mascota_df[columnas_existentes])

# 4. Seleccionar solo las columnas que el modelo espera y en
# el orden correcto
nueva_mascota_df = nueva_mascota_df[X_train.columns]

# Hacer la predicción
prediccion = model.predict(nueva_mascota_df)

if prediccion[0] == 1:
    print("Es probable que la mascota sea adoptada en 30 días.")
else:
    print("Es poco probable que la mascota sea adoptada en 30 días.")