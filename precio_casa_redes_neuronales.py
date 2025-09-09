import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow import keras



# Generar datos de ejemplo (simulando tamaño y habitaciones)
np.random.seed(42)
num_samples = 200
tamaño = np.random.uniform(50, 200, num_samples).reshape(-1, 1)
habitaciones = np.random.randint(1, 5, num_samples).reshape(-1, 1)
precio_base = 50000
precio_por_metro = 2000
precio_por_habitacion = 10000
ruido = np.random.normal(0, 30000, num_samples).reshape(-1, 1) # Añadir ruido para hacerlo más realista
precio = precio_base + precio_por_metro * tamaño + precio_por_habitacion * habitaciones + ruido

# Combinar características
X = np.concatenate((tamaño, habitaciones), axis=1)
y = precio

# Escalar las características (importante para redes neuronales)
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Escalar la variable objetivo (opcional, pero puede ayudar a la convergencia)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

print("Forma de los datos de entrenamiento (características):", X_train.shape)
print("Forma de los datos de entrenamiento (precio):", y_train.shape)
print("Forma de los datos de prueba (características):", X_test.shape)
print("Forma de los datos de prueba (precio):", y_test.shape)

model_regression = keras.Sequential([
    keras.layers.Dense(units=32, activation='relu', input_shape=(2,)), # Capa oculta con 64 neuronas
    keras.layers.Dense(units=16, activation='relu'),                 # Otra capa oculta
    keras.layers.Dense(units=1)                                    # Capa de salida con 1 neurona (para predecir el precio)
])

# Compilar el modelo para regresión
model_regression.compile(optimizer='adam', loss='mse', metrics=['mae']) # Mean Squared Error es común para regresión, Mean Absolute Error para interpretabilidad

# Entrenar el modelo
history_regression = model_regression.fit(X_train, y_train, epochs=40, batch_size=16, validation_split=0.1, verbose=0)

# Evaluar el modelo
loss_regression, mae_regression = model_regression.evaluate(X_test, y_test, verbose=0)
print(f"Pérdida (MSE) en el conjunto de prueba: {loss_regression:.4f}")
print(f"Error Absoluto Medio (MAE) en el conjunto de prueba: {mae_regression:.4f}")

# import matplotlib.pyplot as plt
# Error cuadrático (MSE)
plt.plot(history_regression.history['loss'], label='Entrenamiento')
plt.plot(history_regression.history['val_loss'], label='Validación')
plt.xlabel('Epocas')
plt.ylabel('MSE (pérdida)')
plt.legend()
plt.title('Curva de pérdida (MSE)')
plt.show()

# Error Absoluto (MAE)
plt.plot(history_regression.history['mae'], label='Entrenamiento')
plt.plot(history_regression.history['val_mae'], label='Validación')
plt.xlabel('Epocas')
plt.ylabel('MAE')
plt.legend()
plt.title('Curva de error absoluto (MAE)')
plt.show()

# Para hacer una predicción para una casa nueva (ejemplo: tamaño=100, habitaciones=3)
nueva_casa = np.array([[100, 3]])
nueva_casa_escalada = scaler_X.transform(nueva_casa)
precio_predicho_escalado = model_regression.predict(nueva_casa_escalada)

# Desescalar la predicción para obtener el precio en la escala original
precio_predicho = scaler_y.inverse_transform(precio_predicho_escalado)
print(f"Precio predicho para una casa de 100 m² y 3 habitaciones: ${precio_predicho.flatten()[0]:,.2f}")