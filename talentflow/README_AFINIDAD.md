# Sistema de Evaluación de Afinidad Laboral 🎯

Sistema de Machine Learning y PLN que evalúa la afinidad entre ofertas de trabajo y hojas de vida de candidatos, proporcionando un puntaje de 0 a 10.

## 📋 Descripción

Este proyecto utiliza **Deep Learning** y **Procesamiento de Lenguaje Natural (PLN)** para:
- Analizar descripciones de trabajos y hojas de vida
- Extraer características relevantes usando **TF-IDF**
- Predecir afinidad laboral mediante **redes neuronales**
- Proporcionar interpretaciones y rankings de candidatos

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      SISTEMA DE AFINIDAD                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Entrada:                                                    │
│  • Descripción del trabajo (texto libre)                    │
│  • Hoja de vida del candidato (texto libre)                 │
│                                                              │
│  Procesamiento:                                              │
│  1. Preprocesamiento de texto                               │
│  2. Vectorización con TF-IDF (n-gramas)                     │
│  3. Red neuronal profunda (256→128→64→32→1)                │
│                                                              │
│  Salida:                                                     │
│  • Puntaje de afinidad: 0-10 (decimal)                     │
│  • Interpretación cualitativa                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Características

✅ **Generación de Dataset Sintético**: Crea datos realistas para entrenamiento  
✅ **Preprocesamiento de Texto**: Limpieza y normalización  
✅ **Vectorización TF-IDF**: Extracción de características con n-gramas  
✅ **Red Neuronal Profunda**: Arquitectura optimizada con Dropout y BatchNormalization  
✅ **Métricas Completas**: MAE, MSE, RMSE, R²  
✅ **Visualizaciones**: Curvas de entrenamiento y análisis de predicciones  
✅ **Interfaz Interactiva**: Menú para evaluaciones individuales y en lote  
✅ **Interpretación Automática**: Recomendaciones basadas en puntajes  

## 📦 Instalación

### 1. Clonar el repositorio
```bash
cd d:\Dev\Bootcamp
```

### 2. Instalar dependencias
```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib nltk
```

### 3. Descargar recursos de NLTK (automático en primera ejecución)
El sistema descarga automáticamente los recursos necesarios de NLTK.

## 📊 Estructura de Archivos

```
📁 Bootcamp/
├── 📄 job_affinity_dataset.py       # Generador de dataset sintético
├── 📄 job_affinity_model.py         # Modelo de Deep Learning
├── 📄 job_affinity_predictor.py     # Interfaz para predicciones
├── 📄 job_affinity_dataset.csv      # Dataset generado (auto-creado)
├── 📄 job_affinity_model.h5         # Modelo entrenado (auto-creado)
├── 📄 vectorizers.pkl               # Vectorizadores TF-IDF (auto-creado)
├── 📄 training_history.png          # Gráficos de entrenamiento
├── 📄 predictions_analysis.png      # Análisis de predicciones
└── 📄 README_AFINIDAD.md            # Este archivo
```

## 🎯 Uso del Sistema

### Paso 1: Generar el Dataset

```bash
python job_affinity_dataset.py
```

**Salida esperada:**
- Archivo `job_affinity_dataset.csv` con 2000 muestras
- Distribución balanceada de afinidades
- Ejemplos impresos en consola

**Características del dataset:**
- **job_description**: Descripción completa del puesto
- **resume**: Hoja de vida del candidato
- **affinity_score**: Puntaje objetivo (0-10)

### Paso 2: Entrenar el Modelo

```bash
python job_affinity_model.py
```

**Proceso de entrenamiento:**
1. Carga y preprocesa el dataset
2. Vectoriza textos con TF-IDF (500 features máx.)
3. Entrena red neuronal (100 épocas máx. con early stopping)
4. Evalúa en conjunto de prueba
5. Genera visualizaciones
6. Guarda modelo y vectorizadores

**Resultados esperados:**
- MAE (Error Absoluto Medio): < 1.0
- R² Score: > 0.70
- 60%+ de predicciones con error ≤ 1.0

### Paso 3: Usar el Modelo para Predicciones

```bash
python job_affinity_predictor.py
```

**Opciones disponibles:**

#### 1️⃣ Evaluar un candidato individual
Ingresa una descripción de trabajo y un CV para obtener la afinidad.

**Ejemplo:**
```
TRABAJO:
Puesto: Desarrollador Full Stack. Ubicación: Remoto. 
Experiencia: 3-5 años. Skills: Python, Django, React, PostgreSQL.

CV:
Experiencia: 4 años. Skills: Python, Django, React, Node.js, PostgreSQL.

RESULTADO: 8.5/10 - Excelente candidato
```

#### 2️⃣ Evaluar múltiples candidatos
Compara varios candidatos para el mismo puesto y obtén un ranking.

#### 3️⃣ Evaluación en lote desde CSV
Procesa múltiples evaluaciones desde un archivo CSV.

**Formato del CSV de entrada:**
```csv
job_description,resume
"Puesto: Desarrollador...","Experiencia: 5 años..."
"Puesto: Data Scientist...","Experiencia: 3 años..."
```

**Salida:** `evaluation_results.csv` con puntajes y interpretaciones

#### 4️⃣ Ver ejemplos
Muestra casos de uso pre-configurados con diferentes perfiles.

## 🧠 Arquitectura del Modelo

### Red Neuronal

```python
Sequential([
    Dense(256, activation='relu')     # Capa densa con 256 neuronas
    Dropout(0.3)                       # Regularización
    Dense(128, activation='relu')     # Capa oculta
    BatchNormalization()               # Normalización
    Dropout(0.3)
    Dense(64, activation='relu')
    BatchNormalization()
    Dropout(0.2)
    Dense(32, activation='relu')
    Dropout(0.2)
    Dense(1, activation='linear')      # Salida: valor continuo 0-10
])
```

### Características del Modelo

- **Entrada**: Vectores TF-IDF concatenados (job + resume)
- **Dimensión de entrada**: ~1000 features (500 por cada texto)
- **Función de pérdida**: MSE (Mean Squared Error)
- **Optimizador**: Adam con learning rate adaptativo
- **Regularización**: Dropout y BatchNormalization
- **Callbacks**: Early Stopping y ReduceLROnPlateau

## 📈 Interpretación de Puntajes

| Puntaje | Interpretación | Recomendación |
|---------|----------------|---------------|
| 8.5 - 10.0 | 🟢 Excelente | Candidato altamente calificado - Entrevista prioritaria |
| 7.0 - 8.4 | 🟢 Muy Bueno | Muy adecuado para el puesto - Considerar para entrevista |
| 5.5 - 6.9 | 🟡 Bueno | Candidato con potencial - Evaluar con más detalle |
| 4.0 - 5.4 | 🟡 Aceptable | Algunas carencias - Puede requerir capacitación |
| 2.5 - 3.9 | 🔴 Bajo | Poco adecuado - Considerar otros candidatos |
| 0.0 - 2.4 | 🔴 Muy Bajo | No cumple requisitos básicos |

## 🔍 Técnicas Implementadas

### Preprocesamiento de Texto
- Conversión a minúsculas
- Eliminación de caracteres especiales
- Preservación de acentos en español
- Tokenización

### Extracción de Características
- **TF-IDF Vectorization**: Identifica palabras relevantes
- **N-gramas (1,2)**: Captura unigramas y bigramas
- **Vocabulario limitado**: 500 features por texto
- **Min DF = 2**: Elimina términos muy raros

### Técnicas de Regularización
- **Dropout (0.2-0.3)**: Previene overfitting
- **Batch Normalization**: Acelera convergencia
- **Early Stopping**: Detiene entrenamiento óptimo
- **Learning Rate Reduction**: Ajusta tasa de aprendizaje

### Evaluación del Modelo
- **MAE**: Error absoluto promedio
- **RMSE**: Raíz del error cuadrático
- **R² Score**: Bondad de ajuste
- **Análisis de errores**: Distribución y rangos

## 🎨 Visualizaciones

### 1. Curva de Entrenamiento
Muestra la pérdida (MSE) y MAE durante el entrenamiento y validación.

### 2. Predicciones vs Valores Reales
Scatter plot que compara predicciones con valores verdaderos.

### 3. Distribución de Errores
Histograma de errores para identificar sesgos.

## 🔧 Configuración Avanzada

### Ajustar el Modelo

En `job_affinity_model.py`, puedes modificar:

```python
# Cambiar número de features TF-IDF
model = JobAffinityModel(max_features=1000)  # Default: 500

# Ajustar arquitectura de red
model.model = keras.Sequential([
    layers.Dense(512, activation='relu'),  # Más neuronas
    # ... agregar más capas
])

# Modificar hiperparámetros de entrenamiento
history = model.train(
    X_train, y_train,
    epochs=200,           # Más épocas
    batch_size=16,        # Batch size menor
    validation_split=0.3  # Más datos de validación
)
```

### Generar Más Datos

En `job_affinity_dataset.py`:

```python
# Generar dataset más grande
df = generate_dataset(num_samples=5000)  # Default: 2000

# Agregar más skills, posiciones, etc. a los catálogos
SKILLS['tech'].extend(['Rust', 'Go', 'Swift'])
```

## 📝 Formato de Entrada

### Descripción de Trabajo (Recomendado)

```
Puesto: [Nombre del puesto]
Ubicación: [Remoto/Ciudad/Híbrido]
Experiencia requerida: [X-Y años]
Educación: [Nivel educativo]
Skills técnicas requeridas: [Habilidad 1, Habilidad 2, ...]
Skills blandas: [Habilidad 1, Habilidad 2, ...]
Idiomas: [Idioma y nivel]
```

### Hoja de Vida (Recomendado)

```
Experiencia: [X-Y años o descripción]
Educación: [Títulos y certificaciones]
Skills: [Habilidad 1, Habilidad 2, ...]
Idiomas: [Idioma 1, Idioma 2, ...]
```

**Nota**: El modelo es flexible y puede trabajar con textos libres, pero seguir este formato mejora la precisión.

## 🐛 Solución de Problemas

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### Error: "Model file not found"
Asegúrate de ejecutar primero:
1. `python job_affinity_dataset.py`
2. `python job_affinity_model.py`

### Predicciones inexactas
- Genera más datos de entrenamiento (aumentar `num_samples`)
- Ajusta la arquitectura del modelo
- Aumenta el número de features TF-IDF

### El modelo no converge
- Reduce el learning rate
- Aumenta el número de épocas
- Verifica la calidad de los datos

## 🚀 Mejoras Futuras

### Técnicas Avanzadas de PLN
- [ ] **Word Embeddings**: Word2Vec, GloVe, FastText
- [ ] **Transformers**: BERT, RoBERTa para español
- [ ] **Sentence Embeddings**: Universal Sentence Encoder

### Características Adicionales
- [ ] Análisis de experiencia en años (numérico)
- [ ] Matching geográfico con penalización de distancia
- [ ] Análisis de certificaciones y educación continua
- [ ] Detección de soft skills implícitas

### Funcionalidades
- [ ] API REST para integración
- [ ] Dashboard web interactivo
- [ ] Explicabilidad del modelo (LIME, SHAP)
- [ ] Sistema de recomendación bidireccional

### Dataset Real
- [ ] Integración con LinkedIn API
- [ ] Web scraping de portales de empleo
- [ ] Datasets públicos como Kaggle Jobs

## 📚 Referencias y Recursos

### Datasets Recomendados (Alternativos)

Si deseas usar datos reales en lugar del dataset sintético:

1. **Kaggle - Job Recommendation Challenge**
   - URL: https://www.kaggle.com/c/job-recommendation
   - Contiene: Descripciones de trabajos y perfiles de usuarios

2. **Indeed Job Postings (Kaggle)**
   - URL: https://www.kaggle.com/datasets/promptcloud/indeed-job-posting-dataset
   - Contiene: 20K+ descripciones de trabajos reales

3. **LinkedIn Job Postings (GitHub)**
   - Buscar: "linkedin job postings dataset github"
   - Múltiples repositorios con datos scrapeados

4. **CareerBuilder Job Recommendation (Kaggle)**
   - Datos de recomendaciones laborales con perfiles

### Tecnologías Utilizadas

- **TensorFlow/Keras**: Deep Learning framework
- **scikit-learn**: Machine Learning y preprocesamiento
- **NLTK**: Procesamiento de lenguaje natural
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas
- **Matplotlib**: Visualizaciones

### Papers Relacionados

- "Deep Learning for Job Recommendation" - RecSys
- "Neural Collaborative Filtering" - WWW 2017
- "BERT for Job Matching" - EMNLP 2020

## 👥 Contribuciones

Este es un proyecto educativo. Sugerencias de mejora:

1. Fork el repositorio
2. Crea una rama con tu feature
3. Implementa mejoras
4. Documenta cambios
5. Crea un Pull Request

## 📄 Licencia

Proyecto educativo - Uso libre para aprendizaje

## 💡 Consejos de Uso

### Para Reclutadores
1. Usa el sistema como **filtro inicial**, no como decisión final
2. Combina el puntaje con entrevistas y otras evaluaciones
3. Considera el contexto cultural y geográfico
4. Revisa candidatos con puntajes > 6.0

### Para Candidatos
1. Optimiza tu CV incluyendo **keywords relevantes**
2. Menciona tecnologías y skills específicas
3. Cuantifica tu experiencia (años, proyectos)
4. Incluye soft skills explícitamente

### Para Desarrolladores
1. Experimenta con diferentes arquitecturas
2. Prueba técnicas avanzadas de PLN
3. Evalúa con métricas de negocio (no solo técnicas)
4. Considera el trade-off precisión vs. interpretabilidad

---

**Desarrollado con ❤️ usando Python, TensorFlow y técnicas de Machine Learning**

Para preguntas o soporte, revisar el código fuente comentado en los archivos .py
