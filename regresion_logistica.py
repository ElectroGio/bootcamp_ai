import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

X = np.array([[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1,1])  

#Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Crear el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Realizar predicciones
probabilidad_aprobar = model.predict_proba([[10]])[0,1]
print(f"Probabilidad de aprobar: {probabilidad_aprobar}")


if probabilidad_aprobar >= 0.69:
    aprueba = 'Si'
else:
    aprueba = 'No'

y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión: {accuracy:.2f}")
