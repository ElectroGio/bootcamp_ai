import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Datos de ejemplo (tamaño de la casa, precio)
X = np.array([100, 150, 200, 250, 300]).reshape((-1, 1)) # Reshape para que sea una matriz 2D
print(X)
y = np.array([200000, 300000, 400000, 500000, 600000])

# Crear un modelo de Regresión Lineal
model = LinearRegression()
# Entrenar el modelo
model.fit(X, y)

# Predecir el precio para una casa de 220 metros cuadrados
valor_predecir = 220
precio_predicho = model.predict([[valor_predecir]])
print(f"Precio predicho para una casa de 220 m2: {precio_predicho[0]:.2f}")

# Visualizar la línea de regresión
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, model.predict(X), color='red', label='Regresión Lineal')
plt.scatter([valor_predecir], precio_predicho, color='green', label=f'Predicción para {valor_predecir} m2', zorder=5)
plt.xlabel('Tamaño de la casa (m2)')
plt.ylabel('Precio')
plt.legend()
plt.show()
