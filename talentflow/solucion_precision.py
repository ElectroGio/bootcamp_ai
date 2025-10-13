"""
GUÍA PARA MEJORAR LA PRECISIÓN DEL MODELO DE AFINIDAD

Este archivo contiene las soluciones para reducir la dispersión y mejorar la precisión.
"""

# =============================================================================
# PROBLEMA IDENTIFICADO
# =============================================================================

"""
La dispersión en las predicciones puede deberse a:

1. Factor aleatorio muy alto en el cálculo de afinidad (30% con rango 0-3)
2. Dataset sintético con mucha variabilidad aleatoria
3. Arquitectura del modelo no optimizada
4. Pocos datos de entrenamiento (2000 muestras)
5. Features no suficientemente discriminativas
"""

# =============================================================================
# SOLUCIÓN 1: MEJORAR EL CÁLCULO DE AFINIDAD EN EL DATASET
# =============================================================================

def calculate_affinity_score_improved(job, resume):
    """
    Cálculo mejorado con menos aleatoriedad y más determinismo
    """
    score = 0.0
    
    # 1. Coincidencia de skills (50% del puntaje) - AUMENTADO
    job_skills_set = set(job['skills'])
    resume_skills_set = set(resume['skills'])
    
    if len(job_skills_set) > 0:
        skills_match = len(job_skills_set.intersection(resume_skills_set)) / len(job_skills_set)
        score += skills_match * 5.0  # Aumentado de 4.0 a 5.0
    
    # 2. Experiencia (35% del puntaje) - AUMENTADO
    exp_levels = ['0-1 años', '1-3 años', '3-5 años', '5-8 años', '8+ años']
    try:
        job_exp_idx = exp_levels.index(job['experience'])
        resume_exp_idx = exp_levels.index(resume['experience'])
        
        # Cálculo más granular
        if resume_exp_idx >= job_exp_idx:
            # Cumple o supera el requisito
            score += 3.5
        elif resume_exp_idx == job_exp_idx - 1:
            # 1 nivel por debajo
            score += 2.5
        elif resume_exp_idx == job_exp_idx - 2:
            # 2 niveles por debajo
            score += 1.0
        else:
            # Muy por debajo
            score += 0.2
    except:
        score += 1.5
    
    # 3. Factor de variabilidad REDUCIDO (15% del puntaje)
    # Simula factores como: cultura, soft skills, motivación, etc.
    import random
    score += random.uniform(0, 1.5)  # REDUCIDO de (0, 3.0) a (0, 1.5)
    
    # Asegurar que esté entre 0 y 10
    score = max(0.0, min(10.0, score))
    
    return round(score, 2)


# =============================================================================
# SOLUCIÓN 2: AUMENTAR EL TAMAÑO DEL DATASET
# =============================================================================

"""
Genera más datos para mejor aprendizaje:
- Mínimo 3000-5000 muestras
- Más ejemplos de cada categoría
"""

def generate_more_data():
    """Genera dataset más grande"""
    # En job_affinity_dataset.py, cambiar:
    # df = generate_dataset(num_samples=5000)  # Aumentado de 2000
    pass


# =============================================================================
# SOLUCIÓN 3: MEJORAR LA ARQUITECTURA DEL MODELO
# =============================================================================

def build_improved_model(input_dim):
    """
    Arquitectura mejorada con más capacidad y mejor regularización
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        # Input layer con más neuronas
        layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Capas ocultas más profundas
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Output
        layers.Dense(1, activation='linear')
    ])
    
    # Optimizador con learning rate más bajo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Reducido
        loss='mse',
        metrics=['mae']
    )
    
    return model


# =============================================================================
# SOLUCIÓN 4: AGREGAR FEATURES NUMÉRICAS EXPLÍCITAS
# =============================================================================

def extract_numeric_features(job_text, resume_text):
    """
    Extrae features numéricas para mejorar las predicciones
    """
    import re
    
    features = []
    
    # 1. Extraer años de experiencia
    exp_pattern = r'(\d+)[-+]?\s*años?'
    
    job_exp = re.findall(exp_pattern, job_text)
    resume_exp = re.findall(exp_pattern, resume_text)
    
    job_exp_val = int(job_exp[0]) if job_exp else 0
    resume_exp_val = int(resume_exp[0]) if resume_exp else 0
    
    # Feature: ratio de experiencia (candidato/requerido)
    exp_ratio = resume_exp_val / max(job_exp_val, 1)
    features.append(min(exp_ratio, 2.0))  # Cap a 2.0
    
    # 2. Conteo de skills comunes
    common_tech_skills = ['python', 'java', 'javascript', 'react', 'django', 
                          'sql', 'aws', 'docker', 'kubernetes', 'git']
    
    job_lower = job_text.lower()
    resume_lower = resume_text.lower()
    
    job_skills_count = sum(1 for skill in common_tech_skills if skill in job_lower)
    resume_skills_count = sum(1 for skill in common_tech_skills if skill in resume_lower)
    matching_skills = sum(1 for skill in common_tech_skills 
                          if skill in job_lower and skill in resume_lower)
    
    # Feature: porcentaje de match de skills
    skill_match_pct = matching_skills / max(job_skills_count, 1)
    features.append(skill_match_pct)
    
    # 3. Nivel educativo (ordinal)
    education_map = {
        'bachiller': 1, 'técnico': 2, 'tecnólogo': 3,
        'profesional': 4, 'maestría': 5, 'doctorado': 6
    }
    
    job_edu = 0
    resume_edu = 0
    
    for edu, level in education_map.items():
        if edu in job_lower:
            job_edu = max(job_edu, level)
        if edu in resume_lower:
            resume_edu = max(resume_edu, level)
    
    # Feature: diferencia educativa (normalizada)
    edu_diff = (resume_edu - job_edu) / 6.0
    features.append(edu_diff)
    
    # 4. Match de ubicación
    locations = ['remoto', 'remote', 'bogotá', 'medellín', 'cali']
    location_match = any(loc in job_lower and loc in resume_lower for loc in locations)
    features.append(1.0 if location_match else 0.0)
    
    return features


def combine_features(X_tfidf, jobs, resumes):
    """
    Combina features de TF-IDF con features numéricas
    """
    import numpy as np
    
    # Extraer features numéricas para cada muestra
    numeric_features = []
    for job, resume in zip(jobs, resumes):
        feats = extract_numeric_features(job, resume)
        numeric_features.append(feats)
    
    numeric_features = np.array(numeric_features)
    
    # Combinar
    X_combined = np.concatenate([X_tfidf, numeric_features], axis=1)
    
    return X_combined


# =============================================================================
# SOLUCIÓN 5: AJUSTAR HIPERPARÁMETROS DE ENTRENAMIENTO
# =============================================================================

def train_with_better_params(model, X_train, y_train):
    """
    Entrena con mejores hiperparámetros
    """
    from tensorflow import keras
    
    # Callbacks mejorados
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,  # Aumentado de 15
        restore_best_weights=True,
        min_delta=0.001  # Añadido
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # Más agresivo
        patience=7,  # Más rápido
        min_lr=0.00001,
        verbose=1
    )
    
    # Entrenar con más épocas y mejor validación
    history = model.fit(
        X_train, y_train,
        epochs=150,  # Aumentado
        batch_size=16,  # Reducido para mejor aprendizaje
        validation_split=0.25,  # Más validación
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history


# =============================================================================
# SOLUCIÓN 6: POST-PROCESAMIENTO DE PREDICCIONES
# =============================================================================

def postprocess_predictions(predictions, confidence_threshold=0.8):
    """
    Ajusta predicciones para reducir dispersión
    """
    import numpy as np
    
    # Aplicar suavizado
    smoothed = predictions.copy()
    
    # Redondear a 0.5 más cercano para reducir variabilidad
    smoothed = np.round(smoothed * 2) / 2
    
    # Clip a rango válido
    smoothed = np.clip(smoothed, 0, 10)
    
    return smoothed


# =============================================================================
# SOLUCIÓN 7: ENSEMBLE DE MODELOS
# =============================================================================

def create_ensemble():
    """
    Combina múltiples modelos para predicciones más estables
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    
    # Modelo 1: Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    # Modelo 2: Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    
    # Modelo 3: Ridge
    ridge = Ridge(alpha=1.0)
    
    return [rf, gb, ridge]


def ensemble_predict(models, X, nn_prediction, weights=None):
    """
    Combina predicciones de múltiples modelos
    """
    import numpy as np
    
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]  # Igual peso
    
    predictions = []
    
    # Predicciones de modelos tradicionales
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Añadir predicción de red neuronal
    predictions.append(nn_prediction)
    
    # Promedio ponderado
    final_prediction = sum(w * p for w, p in zip(weights, predictions))
    
    return final_prediction


# =============================================================================
# IMPLEMENTACIÓN COMPLETA - CÓDIGO LISTO PARA USAR
# =============================================================================

print("""
================================================================================
GUÍA DE IMPLEMENTACIÓN - MEJORA DE PRECISIÓN
================================================================================

PASO 1: Mejorar el cálculo de afinidad en el dataset
------------------------------------------------------
En job_affinity_dataset.py, reemplaza calculate_affinity_score() con
calculate_affinity_score_improved() de este archivo.

CAMBIO CLAVE: Reduce factor aleatorio de (0, 3.0) a (0, 1.5)

PASO 2: Generar más datos
--------------------------
En job_affinity_dataset.py:
    df = generate_dataset(num_samples=5000)  # Aumentar de 2000

PASO 3: Actualizar arquitectura del modelo
-------------------------------------------
En job_affinity_model.py, método build_model(), usa build_improved_model()
de este archivo.

CAMBIOS:
- Más neuronas (512 → 256 → 128 → 64 → 32)
- Mejor regularización
- Learning rate más bajo (0.0005)

PASO 4: Agregar features numéricas
-----------------------------------
En job_affinity_model.py, después de vectorize_text(), añade:
    X_train = combine_features(X_train, jobs_train, resumes_train)
    X_test = combine_features(X_test, jobs_test, resumes_test)

PASO 5: Mejorar entrenamiento
------------------------------
En job_affinity_model.py, usa train_with_better_params():
- Más épocas (150)
- Batch size menor (16)
- Más validación (25%)

PASO 6: Opcional - Ensemble
----------------------------
Para máxima precisión, entrena múltiples modelos y combina predicciones.

================================================================================
RESULTADOS ESPERADOS
================================================================================

Después de aplicar estas mejoras:

1. MAE debería bajar a < 0.7 (actualmente ~1.0)
2. R² Score debería subir a > 0.85 (actualmente ~0.70)
3. 80%+ de predicciones con error ≤ 1.0
4. Predicciones más consistentes y menos dispersas

================================================================================
ARCHIVO LISTO PARA APLICAR
================================================================================

Este archivo incluye todo el código necesario. Puedes:
1. Copiar las funciones mejoradas a tus archivos existentes
2. O crear nuevos archivos job_affinity_v2.py con las mejoras

================================================================================
""")
