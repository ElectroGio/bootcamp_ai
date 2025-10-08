"""
RECOMENDACIONES Y MEJORAS AVANZADAS
Sistema de Evaluación de Afinidad Laboral
"""

# ============================================================================
# 📊 RECOMENDACIONES PARA MEJORAR EL MODELO
# ============================================================================

"""
1. USAR EMBEDDINGS PRE-ENTRENADOS
   ================================
   
   Alternativa al TF-IDF: Usar embeddings de palabras pre-entrenados
   que capturan mejor el contexto semántico.
   
   Opciones:
   a) Word2Vec / FastText en español
   b) BERT en español (BETO, RoBERTa-es)
   c) Universal Sentence Encoder
   
   Ejemplo con Sentence Transformers:
   
   from sentence_transformers import SentenceTransformer
   
   model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
   job_embeddings = model.encode(job_descriptions)
   resume_embeddings = model.encode(resumes)
"""

# Ejemplo de implementación con Sentence Transformers
def ejemplo_sentence_transformers():
    """
    Mejora: Usar embeddings de oraciones en lugar de TF-IDF
    Ventajas: Captura contexto semántico, mejor para textos largos
    """
    
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    # Cargar modelo pre-entrenado multilingüe
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # Ejemplo
    job = "Desarrollador Python con experiencia en Django y React"
    resume = "5 años de experiencia con Python, Django, React y AWS"
    
    # Generar embeddings (vectores de 768 dimensiones)
    job_embedding = model.encode(job)
    resume_embedding = model.encode(resume)
    
    # Calcular similitud coseno
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]
    
    print(f"Similitud semántica: {similarity:.4f}")
    
    return job_embedding, resume_embedding


"""
2. USAR DATASETS REALES
   =====================
   
   Para producción, considera usar datasets reales:
   
   a) Kaggle - Job Recommendation Challenge
      https://www.kaggle.com/c/job-recommendation
   
   b) LinkedIn Public Dataset (GitHub)
      Buscar: "linkedin job postings dataset"
   
   c) Indeed Job Postings
      https://www.kaggle.com/datasets/promptcloud/indeed-job-posting-dataset
   
   d) Stack Overflow Developer Survey
      https://insights.stackoverflow.com/survey
"""

def cargar_dataset_real():
    """
    Ejemplo de cómo cargar y adaptar un dataset real
    """
    import pandas as pd
    
    # Ejemplo con dataset de Kaggle
    # df = pd.read_csv('path/to/kaggle_jobs.csv')
    
    # Adaptar columnas al formato esperado
    # df_adapted = pd.DataFrame({
    #     'job_description': df['description'] + ' ' + df['requirements'],
    #     'resume': df['user_profile'],
    #     'affinity_score': df['match_score'] * 10  # Normalizar a 0-10
    # })
    
    pass


"""
3. ANÁLISIS DE SKILLS ESPECÍFICAS
   ================================
   
   Mejora: Extraer y comparar skills de forma explícita
   usando Named Entity Recognition (NER) o diccionarios
"""

def extraer_skills_avanzado():
    """
    Extracción avanzada de skills usando spaCy o regex
    """
    import re
    
    # Diccionario de skills conocidas
    TECH_SKILLS = {
        'lenguajes': ['Python', 'Java', 'JavaScript', 'C++', 'Ruby', 'Go', 'Rust'],
        'frameworks': ['Django', 'Flask', 'React', 'Angular', 'Vue', 'Spring'],
        'databases': ['PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Cassandra'],
        'cloud': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes'],
        'ml': ['TensorFlow', 'PyTorch', 'Scikit-learn', 'Keras']
    }
    
    def extract_skills(text):
        """Extrae skills mencionadas en el texto"""
        text_lower = text.lower()
        found_skills = []
        
        for category, skills in TECH_SKILLS.items():
            for skill in skills:
                if skill.lower() in text_lower:
                    found_skills.append({
                        'skill': skill,
                        'category': category
                    })
        
        return found_skills
    
    # Ejemplo
    job = "Buscamos desarrollador con Python, Django y AWS"
    resume = "Experiencia en Python, Flask y Docker"
    
    job_skills = extract_skills(job)
    resume_skills = extract_skills(resume)
    
    print(f"Skills requeridas: {job_skills}")
    print(f"Skills del candidato: {resume_skills}")
    
    # Calcular overlap
    job_skill_names = {s['skill'] for s in job_skills}
    resume_skill_names = {s['skill'] for s in resume_skills}
    
    overlap = len(job_skill_names & resume_skill_names) / len(job_skill_names)
    print(f"Overlap de skills: {overlap:.2%}")


"""
4. ARQUITECTURA SIAMESE NETWORK
   ==============================
   
   Mejora: Usar redes siamesas para aprender embeddings
   optimizados para comparación de trabajos vs CVs
"""

def ejemplo_siamese_network():
    """
    Arquitectura de red siamesa para matching
    Ventaja: Aprende representaciones óptimas para comparación
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    
    def crear_encoder(input_dim):
        """Crea el encoder compartido"""
        return keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu')
        ])
    
    # Inputs
    job_input = layers.Input(shape=(input_dim,), name='job_input')
    resume_input = layers.Input(shape=(input_dim,), name='resume_input')
    
    # Encoder compartido
    encoder = crear_encoder(input_dim)
    
    # Codificar ambos inputs
    job_encoded = encoder(job_input)
    resume_encoded = encoder(resume_input)
    
    # Concatenar o calcular distancia
    merged = layers.concatenate([job_encoded, resume_encoded])
    
    # Capas de decisión
    x = layers.Dense(32, activation='relu')(merged)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation='linear')(x)  # Afinidad 0-10
    
    # Modelo
    model = keras.Model(
        inputs=[job_input, resume_input],
        outputs=output,
        name='siamese_matcher'
    )
    
    return model


"""
5. FEATURE ENGINEERING AVANZADO
   ==============================
   
   Agregar características numéricas estructuradas
"""

def feature_engineering_avanzado():
    """
    Extrae features numéricas adicionales para mejorar predicciones
    """
    import re
    
    def extraer_features_numericas(job, resume):
        """
        Extrae features cuantitativas del texto
        """
        features = {}
        
        # 1. Longitud de textos
        features['job_length'] = len(job.split())
        features['resume_length'] = len(resume.split())
        
        # 2. Experiencia en años (regex)
        exp_pattern = r'(\d+)[-+]?\s*(?:años?|years?)'
        
        job_exp = re.findall(exp_pattern, job.lower())
        resume_exp = re.findall(exp_pattern, resume.lower())
        
        features['job_min_exp'] = int(job_exp[0]) if job_exp else 0
        features['resume_exp'] = int(resume_exp[0]) if resume_exp else 0
        
        # 3. Nivel de educación (ordinal)
        education_levels = {
            'bachiller': 1, 'técnico': 2, 'tecnólogo': 3,
            'profesional': 4, 'maestría': 5, 'doctorado': 6
        }
        
        features['job_edu_level'] = 0
        features['resume_edu_level'] = 0
        
        for edu, level in education_levels.items():
            if edu in job.lower():
                features['job_edu_level'] = max(features['job_edu_level'], level)
            if edu in resume.lower():
                features['resume_edu_level'] = max(features['resume_edu_level'], level)
        
        # 4. Nivel de inglés (ordinal)
        english_levels = {
            'básico': 1, 'basic': 1,
            'intermedio': 2, 'intermediate': 2,
            'avanzado': 3, 'advanced': 3,
            'nativo': 4, 'native': 4
        }
        
        features['resume_english'] = 0
        for level_text, level_val in english_levels.items():
            if f'inglés {level_text}' in resume.lower() or f'english {level_text}' in resume.lower():
                features['resume_english'] = max(features['resume_english'], level_val)
        
        # 5. Ubicación match (booleano)
        locations = ['remoto', 'remote', 'bogotá', 'medellín', 'cali', 'híbrido', 'hybrid']
        features['location_match'] = any(loc in job.lower() and loc in resume.lower() for loc in locations)
        
        return features
    
    # Ejemplo
    job = "Puesto en Bogotá, 3-5 años de experiencia, Profesional"
    resume = "5 años de experiencia, Ingeniero, en Bogotá, Inglés avanzado"
    
    features = extraer_features_numericas(job, resume)
    print("Features numéricas:", features)
    
    return features


"""
6. ENSEMBLE MODELS
   ================
   
   Combinar múltiples modelos para mejor precisión
"""

def ejemplo_ensemble():
    """
    Ensemble de múltiples modelos para predicción robusta
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    
    # Modelo 1: Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Modelo 2: Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    # Modelo 3: Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    
    # Modelo 4: Red Neuronal (ya implementado)
    # nn_model = tu_modelo_keras
    
    # Predicción con votación/promedio
    def ensemble_predict(X):
        pred_rf = rf_model.predict(X)
        pred_gb = gb_model.predict(X)
        pred_ridge = ridge_model.predict(X)
        # pred_nn = nn_model.predict(X)
        
        # Promedio ponderado
        ensemble_pred = (0.3 * pred_rf + 0.3 * pred_gb + 
                        0.2 * pred_ridge)  # + 0.2 * pred_nn
        
        return ensemble_pred


"""
7. EXPLICABILIDAD DEL MODELO
   ==========================
   
   Usar LIME o SHAP para explicar predicciones
"""

def ejemplo_explicabilidad():
    """
    Añadir explicabilidad al modelo con SHAP
    """
    # import shap
    
    # # Crear explainer
    # explainer = shap.KernelExplainer(model.predict, X_train_sample)
    
    # # Explicar una predicción
    # shap_values = explainer.shap_values(X_test[0:1])
    
    # # Visualizar
    # shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])
    
    # Alternativa simple: mostrar features más importantes
    def explicar_prediccion_simple(job, resume, affinity):
        """
        Explicación textual simple de la predicción
        """
        explicacion = []
        
        if affinity >= 7.0:
            explicacion.append("✅ Alta coincidencia de skills")
        elif affinity >= 5.0:
            explicacion.append("⚠️ Coincidencia parcial de skills")
        else:
            explicacion.append("❌ Baja coincidencia de skills")
        
        # Analizar experiencia
        if "5-8 años" in resume and "3-5 años" in job:
            explicacion.append("✅ Experiencia supera requerimientos")
        
        # Analizar educación
        if "maestría" in resume.lower() or "doctorado" in resume.lower():
            explicacion.append("✅ Alto nivel educativo")
        
        return "\n".join(explicacion)


"""
8. API REST PARA PRODUCCIÓN
   =========================
   
   Crear API con FastAPI para integración
"""

def ejemplo_api():
    """
    API REST para el sistema de afinidad
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    app = FastAPI(title="Job Affinity API")
    
    class JobRequest(BaseModel):
        job_description: str
        resume: str
    
    class AffinityResponse(BaseModel):
        affinity_score: float
        interpretation: str
        confidence: float
    
    @app.post("/predict", response_model=AffinityResponse)
    async def predict_affinity(request: JobRequest):
        """
        Endpoint para predecir afinidad
        """
        try:
            # Cargar modelo (hacer una vez al inicio)
            # model = load_model()
            
            # Predecir
            # affinity = model.predict_affinity(
            #     request.job_description, 
            #     request.resume
            # )
            
            affinity = 8.5  # Ejemplo
            
            if affinity >= 8.5:
                interpretation = "Excelente"
            elif affinity >= 7.0:
                interpretation = "Muy Bueno"
            else:
                interpretation = "Aceptable"
            
            return AffinityResponse(
                affinity_score=affinity,
                interpretation=interpretation,
                confidence=0.85
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Ejecutar con: uvicorn api:app --reload


"""
9. MÉTRICAS DE NEGOCIO
   ====================
   
   Además de métricas técnicas, evaluar con métricas de negocio
"""

def metricas_negocio():
    """
    Métricas orientadas al negocio
    """
    
    def calcular_metricas_reclutamiento(predicciones, contratados):
        """
        Métricas específicas para reclutamiento
        
        predicciones: array de puntajes predichos
        contratados: array booleano (True si fue contratado)
        """
        
        # 1. Tasa de conversión por rango de afinidad
        rangos = [(8, 10), (6, 8), (4, 6), (0, 4)]
        
        for min_score, max_score in rangos:
            mask = (predicciones >= min_score) & (predicciones < max_score)
            tasa = contratados[mask].mean() if mask.any() else 0
            print(f"Rango {min_score}-{max_score}: {tasa:.2%} contratados")
        
        # 2. Time to hire reduction
        # Candidatos con alta afinidad se procesan primero
        
        # 3. Cost per hire reduction
        # Menos entrevistas innecesarias
        
        # 4. Quality of hire
        # Correlación entre afinidad y desempeño


"""
10. MEJORAS EN EL DATASET
    ======================
    
    Técnicas para mejorar calidad del dataset sintético
"""

def mejorar_dataset_sintetico():
    """
    Mejoras para el generador de dataset
    """
    
    # 1. Usar plantillas más realistas
    PLANTILLAS_DESCRIPCION = [
        "Estamos buscando un {puesto} con experiencia en {skills}. "
        "El candidato ideal tiene {experiencia} años de experiencia y "
        "educación en {educacion}. Ofrecemos ambiente colaborativo y "
        "oportunidades de crecimiento.",
        
        "{puesto} - {ubicacion}. Responsabilidades: {responsabilidades}. "
        "Requisitos: {skills}, {experiencia} de experiencia. "
        "Beneficios: {beneficios}."
    ]
    
    # 2. Agregar ruido y variabilidad
    # - Errores de tipeo intencionales
    # - Sinónimos (developer vs desarrollador)
    # - Diferentes formatos de fecha
    
    # 3. Balancear distribución de afinidad
    # - Asegurar que hay suficientes ejemplos en cada rango
    
    # 4. Casos edge
    # - CVs muy cortos vs muy largos
    # - Trabajos con requisitos vagos
    # - Candidatos sobre-calificados
    
    pass


# ============================================================================
# 📚 RECURSOS ADICIONALES
# ============================================================================

RECURSOS = """
CURSOS Y TUTORIALES:
- Deep Learning Specialization (Coursera - Andrew Ng)
- Natural Language Processing (Hugging Face)
- Applied Machine Learning (Fast.ai)

PAPERS:
- "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- "Neural Collaborative Filtering" (WWW 2017)
- "Wide & Deep Learning for Recommender Systems" (2016)

HERRAMIENTAS:
- Hugging Face Transformers
- spaCy para NLP en español
- Weights & Biases para tracking de experimentos
- MLflow para gestión de modelos

COMUNIDADES:
- r/MachineLearning
- Papers with Code
- Kaggle Competitions
- Stack Overflow
"""

if __name__ == "__main__":
    print("="*80)
    print("RECOMENDACIONES PARA MEJORAR EL SISTEMA DE AFINIDAD LABORAL")
    print("="*80)
    print(RECURSOS)
    print("\n💡 Revisa cada función en este archivo para implementar mejoras")
