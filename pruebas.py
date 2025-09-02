import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Crear un DataFrame de ejemplo
data = {'edad': [25, 30, None, 40, 35],
        'ingresos': [50000, 60000, 70000, 80000, 90000],
        'genero': ['masculino', 'masculino', None, 'femenino', 'masculino']}
df = pd.DataFrame(data)
df.head()
print(df)

# Imputar datos faltantes (edad) con la mediana
imputerNum = SimpleImputer(strategy='median')
imputerObj = SimpleImputer(strategy='most_frequent')
df['edad'] = imputerNum.fit_transform(df[['edad']])
df['genero'] = imputerObj.fit_transform(df[['genero']]).ravel()

print(df)

# Escalar los ingresos entre 0 y 1
scaler = MinMaxScaler()
df['ingresos'] = scaler.fit_transform(df[['ingresos']])

# Crear una copia del DataFrame *antes* de aplicar one-hot encoding (usando el df después de la imputación)
df_manual_encoded = df.copy()

# Codificar la variable género (one-hot encoding)
df = pd.get_dummies(df, columns=['genero'])

print(df)

# Ejemplo 2: Codificación manual de 'genero'

# Definir el mapeo
gender_mapping = {
    'masculino': 1,
    'femenino': 2,
    None: 1}
    # Mapear la cadena vacía a 0 también,
    # ajustar si es necesario

# Aplicar el mapeo
df_manual_encoded['genero_encoded'] = df_manual_encoded['genero'].map(gender_mapping)

# Eliminar la columna original 'genero' si se desea
df_manual_encoded = df_manual_encoded.drop('genero', axis=1)

print("DataFrame con codificación manual:")
print(df_manual_encoded)