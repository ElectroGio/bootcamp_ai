
import tensorflow as tf
from tensorflow import keras

print("Versión de TensorFlow:", tf.__version__)
print("Versión de Keras:", keras.__version__)

# Crear un modelo simple
model = keras.Sequential([
    # Capa densa con 10 neuronas, función de activación ReLU,
    # esperando una entrada de 5 características
    keras.layers.Dense(units=10, activation='relu', input_shape=(5,)),

    # Capa de salida con 1 neurona (para clasificación binaria),
    # función de activación sigmoide
    keras.layers.Dense(units=1, activation='sigmoid')
])

# Muestra la arquitectura del modelo
model.summary()

'''**¿Qué es la función de activación ReLU?**
<p>Ejemplo en la vida real:</p>
<p>Imaginen que tienen un sensor de temperatura en una fábrica.</p>

<p>Si la temperatura es positiva, el sensor envía una alerta.</p>
<p>Si la temperatura es negativa, el sensor la ignora porque solo nos interesan valores positivos.</p>

**Así funciona ReLU:**

* Si la entrada es mayor que 0, la deja pasar sin cambios.
* Si la entrada es menor que 0, la convierte en 0 (ignora valores negativos).

**Fórmula de ReLU:**

ReLU(x)=max(0,x)

**¿Qué es la función de activación Sigmoid?**
<p>Ejemplo en la vida real:</p>
<p>Imagine que un banco quiere aprobar o rechazar una tarjeta de crédito según el historial del cliente:</p>

* Si el cliente tiene buen historial → Alta probabilidad de aprobación (cercano a 1).
* Si el cliente tiene mal historial → Baja probabilidad de aprobación (cercano a 0).

**Así funciona Sigmoid:**

<p>Convierte cualquier número en un valor entre 0 y 1.</p>
<p>Se usa cuando queremos calcular probabilidades.</p>'''