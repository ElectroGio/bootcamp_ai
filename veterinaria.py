import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



# 1. Creación de un conjunto de datos de ejemplo
# En un caso real, cargarías tus datos desde un archivo (por ejemplo, pd.read_csv('tus_datos.csv'))
data = {
    'especie': ['Perro', 'Gato', 'Perro', 'Perro', 'Gato', 'Gato', 'Perro', 'Perro', 'Gato', 'Perro'] * 5,
    'edad': np.random.randint(1, 15, 50),
    'genero': ['Macho', 'Hembra', 'Macho', 'Hembra', 'Macho', 'Hembra', 'Macho', 'Hembra', 'Macho', 'Hembra'] * 5,
    'tamaño': ['Grande', 'Pequeño', 'Mediano', 'Grande', 'Pequeño', 'Pequeño', 'Mediano', 'Grande', 'Pequeño', 'Mediano'] * 5,
    'raza': ['Labrador', 'Siames', 'Mestizo', 'Golden', 'Persa', 'Mestizo', 'Bulldog', 'Mestizo', 'Bengala', 'Poodle'] * 5,
    'esterilizado': np.random.choice([True, False], 50),
    'vacunado': np.random.choice([True, False], 50),
    'necesita_tratamiento_especial': np.random.choice([True, False], 50, p=[0.2, 0.8]),
    'tiempo_en_adopcion': np.random.randint(1, 100, 50),
    'adoptado_en_30_dias': np.random.choice([1, 0], 50) # 1: Si, 0: No
}

data_excel = pd.read_csv('content\data_patitas_felices_v2.csv')


df = pd.DataFrame(data_excel)

print(df.head(20))

#Manejo de datos faltantes.
for column in df.columns:
    if df[column].isnull().any():
        if df[column].dtype == 'object':  # Si es categórica
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:  # Si es numérica
            df[column].fillna(df[column].median(), inplace=True)


print(df.head(20))

scaler = StandardScaler()
df[['edad','tiempo_en_adopcion']] = scaler.fit_transform(df[['edad','tiempo_en_adopcion']])
df.head()


# 2. Preprocesamiento de datos
# Definir las características (X) y la variable objetivo (y)
# 'tiempo_en_adopcion' se excluye porque es información del futuro que no tendríamos al predecir.
features = ['id_mascota','especie', 'edad', 'genero', 'tamaño', 'raza', 'esterilizado', 'vacunado', 'necesita_tratamiento_especial']
#features = df.drop(columns=['adoptado_en_30_dias', 'tiempo_en_adopcion'])

X = df[features]
y = df['adoptado_en_30_dias']

# Identificar características categóricas y numéricas
categorical_features = ['especie', 'genero', 'tamaño', 'raza']
numerical_features = ['edad', 'esterilizado', 'vacunado', 'necesita_tratamiento_especial']

# Crear un transformador para aplicar One-Hot Encoding a las características categóricas
# y pasar las numéricas sin cambios.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 3. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Crear y entrenar el modelo
# Se usará un Pipeline para encadenar el preprocesamiento y el modelo
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(max_iter=1000))])

model.fit(X_train, y_train)

# 5. Realizar predicciones
y_pred = model.predict(X_test)

# 6. Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"--- Evaluación del Modelo de Adopción de Mascotas ---")
print(f"\nPrecisión (Accuracy): {accuracy:.2f}")
print(f"\nMatriz de Confusión:\n{conf_matrix}")
print(f"\nReporte de Clasificación:\n{class_report}")

# 7. Ejemplo de predicción con un nuevo dato
print("\n--- Ejemplo de Predicción ---")
nueva_mascota = pd.DataFrame({
    'especie': ['Perro'],
    'edad': [2],
    'genero': ['Hembra'],
    'tamaño': ['Mediano'],
    'raza': ['Mestizo'],
    'esterilizado': [True],
    'vacunado': [True],
    'necesita_tratamiento_especial': [False]
})

prediccion = model.predict(nueva_mascota)
probabilidad = model.predict_proba(nueva_mascota)

print(f"Datos de la nueva mascota:\n{nueva_mascota.to_string()}")
print(f"\nPredicción: {'Será adoptada en 30 días' if prediccion[0] == 1 else 'No será adoptada en 30 días'}")
print(f"Probabilidades (No/Sí): {probabilidad[0]}")
