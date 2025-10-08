"""
Ejemplos prácticos de uso del Sistema de Afinidad Laboral
Diferentes escenarios y casos de uso reales
"""

# ============================================================================
# EJEMPLO 1: USO BÁSICO - EVALUACIÓN INDIVIDUAL
# ============================================================================

def ejemplo_evaluacion_individual():
    """
    Caso de uso más simple: evaluar un candidato para una vacante
    """
    from job_affinity_model import JobAffinityModel
    import pickle
    
    # Cargar modelo
    print("Cargando modelo...")
    model = JobAffinityModel(max_features=500)
    model.load_model('job_affinity_model.h5')
    
    with open('vectorizers.pkl', 'rb') as f:
        vectorizers = pickle.load(f)
        model.job_vectorizer = vectorizers['job_vectorizer']
        model.resume_vectorizer = vectorizers['resume_vectorizer']
    
    # Datos del trabajo
    job = """
    Puesto: Desarrollador Backend Senior
    Ubicación: Remoto
    Experiencia requerida: 5-8 años
    Educación: Profesional en Ingeniería de Sistemas o afines
    
    Skills técnicas requeridas:
    - Python avanzado
    - Django o Flask
    - PostgreSQL, Redis
    - Docker, Kubernetes
    - AWS (EC2, S3, Lambda)
    - Arquitectura de microservicios
    - APIs REST
    - Git
    
    Skills blandas:
    - Liderazgo técnico
    - Mentoría de desarrolladores junior
    - Trabajo en equipo remoto
    - Comunicación efectiva
    
    Idiomas: Inglés avanzado (mínimo B2)
    
    Beneficios: Salario competitivo, 100% remoto, horario flexible
    """
    
    # CV del candidato
    resume = """
    JUAN PÉREZ
    Desarrollador Backend Senior | 6 años de experiencia
    
    EDUCACIÓN:
    - Ingeniería de Sistemas - Universidad Nacional (2015)
    - Diplomado en Cloud Computing - Coursera (2020)
    
    EXPERIENCIA:
    Backend Developer Senior - TechCorp (2019-2024)
    - Desarrollo de microservicios con Python y Django
    - Diseño e implementación de APIs REST
    - Gestión de bases de datos PostgreSQL y Redis
    - Deploy en AWS con Docker y Kubernetes
    - Liderazgo de equipo de 3 desarrolladores junior
    
    Backend Developer - StartupXYZ (2017-2019)
    - Desarrollo con Flask y PostgreSQL
    - Integración con servicios AWS
    - Implementación de CI/CD con Jenkins
    
    SKILLS TÉCNICAS:
    Python (avanzado), Django, Flask, PostgreSQL, MongoDB, Redis,
    Docker, Kubernetes, AWS (EC2, S3, Lambda, RDS), Git, Jenkins,
    Microservicios, APIs REST, Testing (pytest, unittest)
    
    SKILLS BLANDAS:
    Liderazgo, Mentoría, Trabajo en equipo, Comunicación efectiva,
    Resolución de problemas, Autodidacta
    
    IDIOMAS:
    Español: Nativo
    Inglés: Avanzado (C1) - TOEFL 105/120
    
    CERTIFICACIONES:
    - AWS Solutions Architect Associate
    - Python for Data Science (Coursera)
    """
    
    # Predecir afinidad
    print("\nCalculando afinidad...\n")
    affinity = model.predict_affinity(job, resume)
    
    # Mostrar resultado
    print("="*80)
    print("RESULTADO DE EVALUACIÓN")
    print("="*80)
    print(f"\n📊 PUNTAJE DE AFINIDAD: {affinity}/10\n")
    
    if affinity >= 8.5:
        print("🟢 RECOMENDACIÓN: Candidato EXCELENTE")
        print("   → Agendar entrevista con prioridad alta")
        print("   → Perfil altamente alineado con los requisitos")
    elif affinity >= 7.0:
        print("🟢 RECOMENDACIÓN: Candidato MUY BUENO")
        print("   → Agendar entrevista")
        print("   → Cumple la mayoría de requisitos")
    elif affinity >= 5.5:
        print("🟡 RECOMENDACIÓN: Candidato ACEPTABLE")
        print("   → Considerar para entrevista si hay disponibilidad")
        print("   → Puede requerir capacitación en algunas áreas")
    else:
        print("🔴 RECOMENDACIÓN: Candidato NO cumple requisitos")
        print("   → Considerar otros candidatos")
    
    print("="*80)


# ============================================================================
# EJEMPLO 2: COMPARACIÓN DE MÚLTIPLES CANDIDATOS
# ============================================================================

def ejemplo_ranking_candidatos():
    """
    Evaluar y rankear múltiples candidatos para la misma vacante
    """
    from job_affinity_model import JobAffinityModel
    import pickle
    import pandas as pd
    
    # Cargar modelo
    model = JobAffinityModel(max_features=500)
    model.load_model('job_affinity_model.h5')
    
    with open('vectorizers.pkl', 'rb') as f:
        vectorizers = pickle.load(f)
        model.job_vectorizer = vectorizers['job_vectorizer']
        model.resume_vectorizer = vectorizers['resume_vectorizer']
    
    # Descripción del trabajo
    job = """
    Data Scientist - Modalidad Híbrida
    Bogotá, Colombia
    
    Experiencia: 3-5 años
    Educación: Profesional en Ingeniería, Matemáticas, Estadística o afines
    
    Requisitos:
    - Python (pandas, numpy, scikit-learn)
    - Machine Learning (modelos supervisados y no supervisados)
    - SQL avanzado
    - Visualización (Matplotlib, Seaborn, PowerBI)
    - Estadística aplicada
    
    Deseable:
    - Deep Learning (TensorFlow o PyTorch)
    - Big Data (Spark)
    - Cloud (AWS o Azure)
    
    Inglés: Intermedio
    """
    
    # Candidatos
    candidatos = [
        {
            'nombre': 'María González',
            'cv': """
            Data Scientist con 4 años de experiencia
            Maestría en Estadística
            
            Skills: Python, pandas, numpy, scikit-learn, TensorFlow,
            SQL (PostgreSQL, MySQL), PowerBI, Tableau, Spark, AWS
            
            Experiencia en:
            - Modelos predictivos (regresión, clasificación)
            - Clustering y análisis de componentes principales
            - Deep Learning para NLP
            - Pipelines de datos en AWS
            
            Inglés: Avanzado
            """
        },
        {
            'nombre': 'Carlos Ramírez',
            'cv': """
            Analista de Datos con 2 años de experiencia
            Profesional en Ingeniería Industrial
            
            Skills: Python básico, SQL, Excel avanzado, PowerBI,
            Estadística básica, Tableau
            
            Experiencia en:
            - Análisis descriptivo y dashboards
            - Limpieza de datos
            - Reportería
            
            Inglés: Intermedio
            """
        },
        {
            'nombre': 'Andrea López',
            'cv': """
            Data Scientist con 5 años de experiencia
            Profesional en Matemáticas, Especialización en Analytics
            
            Skills: Python (pandas, numpy, scikit-learn, PyTorch),
            R, SQL avanzado, PowerBI, Machine Learning, Deep Learning,
            Spark, Docker
            
            Experiencia en:
            - Modelos de ML en producción
            - Computer Vision con Deep Learning
            - Análisis estadístico avanzado
            - A/B Testing
            
            Inglés: Intermedio
            Publicaciones en conferencias de ML
            """
        },
        {
            'nombre': 'Roberto Silva',
            'cv': """
            Desarrollador Backend con 3 años de experiencia
            Profesional en Ingeniería de Sistemas
            
            Skills: Python, Django, PostgreSQL, Docker, Git,
            JavaScript básico
            
            Experiencia en:
            - Desarrollo de APIs
            - Backend de aplicaciones web
            
            Inglés: Básico
            """
        }
    ]
    
    # Evaluar cada candidato
    resultados = []
    
    print("\n" + "="*80)
    print("EVALUACIÓN DE CANDIDATOS PARA: Data Scientist")
    print("="*80 + "\n")
    
    for candidato in candidatos:
        affinity = model.predict_affinity(job, candidato['cv'])
        resultados.append({
            'Candidato': candidato['nombre'],
            'Afinidad': affinity,
            'CV': candidato['cv'][:100] + "..."
        })
    
    # Crear DataFrame y ordenar
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values('Afinidad', ascending=False)
    df_resultados['Ranking'] = range(1, len(df_resultados) + 1)
    
    # Mostrar ranking
    print("🏆 RANKING DE CANDIDATOS:\n")
    
    for _, row in df_resultados.iterrows():
        print(f"{row['Ranking']}. {row['Candidato']}")
        print(f"   Afinidad: {row['Afinidad']:.2f}/10")
        
        if row['Afinidad'] >= 8.0:
            print(f"   Estado: 🟢 ENTREVISTAR CON PRIORIDAD")
        elif row['Afinidad'] >= 6.5:
            print(f"   Estado: 🟢 ENTREVISTAR")
        elif row['Afinidad'] >= 5.0:
            print(f"   Estado: 🟡 CONSIDERAR")
        else:
            print(f"   Estado: 🔴 NO CUMPLE REQUISITOS")
        
        print()
    
    print("="*80)
    
    return df_resultados


# ============================================================================
# EJEMPLO 3: PROCESAMIENTO EN LOTE DESDE CSV
# ============================================================================

def ejemplo_batch_processing():
    """
    Procesar múltiples evaluaciones desde un archivo CSV
    """
    import pandas as pd
    from job_affinity_model import JobAffinityModel
    import pickle
    
    # Crear CSV de ejemplo
    print("Creando archivo CSV de ejemplo...")
    
    data = {
        'job_id': ['JOB001', 'JOB001', 'JOB001', 'JOB002', 'JOB002'],
        'job_description': [
            'Desarrollador Frontend - React, JavaScript, CSS',
            'Desarrollador Frontend - React, JavaScript, CSS',
            'Desarrollador Frontend - React, JavaScript, CSS',
            'DevOps Engineer - Docker, Kubernetes, AWS, CI/CD',
            'DevOps Engineer - Docker, Kubernetes, AWS, CI/CD'
        ],
        'candidate_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'resume': [
            '5 años con React, JavaScript, HTML, CSS, Redux',
            '2 años con Angular, TypeScript, básico React',
            '8 años con Backend Python, Django, SQL',
            '4 años DevOps, Docker, Kubernetes, AWS, Jenkins',
            '3 años Sysadmin, Linux, básico Docker'
        ]
    }
    
    df_input = pd.DataFrame(data)
    df_input.to_csv('candidatos_batch.csv', index=False)
    print("✓ Archivo creado: candidatos_batch.csv\n")
    
    # Cargar modelo
    print("Cargando modelo...")
    model = JobAffinityModel(max_features=500)
    model.load_model('job_affinity_model.h5')
    
    with open('vectorizers.pkl', 'rb') as f:
        vectorizers = pickle.load(f)
        model.job_vectorizer = vectorizers['job_vectorizer']
        model.resume_vectorizer = vectorizers['resume_vectorizer']
    
    # Procesar en lote
    print("Procesando evaluaciones...\n")
    
    affinities = []
    for _, row in df_input.iterrows():
        affinity = model.predict_affinity(
            row['job_description'],
            row['resume']
        )
        affinities.append(affinity)
    
    df_input['affinity_score'] = affinities
    
    # Agregar interpretación
    def interpretar(score):
        if score >= 8.5:
            return 'Excelente'
        elif score >= 7.0:
            return 'Muy Bueno'
        elif score >= 5.5:
            return 'Bueno'
        elif score >= 4.0:
            return 'Aceptable'
        else:
            return 'No cumple'
    
    df_input['interpretation'] = df_input['affinity_score'].apply(interpretar)
    df_input['status'] = df_input['affinity_score'].apply(
        lambda x: 'ENTREVISTAR' if x >= 7.0 else 'REVISAR' if x >= 5.5 else 'RECHAZAR'
    )
    
    # Guardar resultados
    df_input.to_csv('resultados_batch.csv', index=False)
    print("✓ Resultados guardados en: resultados_batch.csv\n")
    
    # Mostrar resumen
    print("="*80)
    print("RESUMEN DE PROCESAMIENTO EN LOTE")
    print("="*80)
    print(f"\nTotal de evaluaciones: {len(df_input)}")
    print(f"Puntaje promedio: {df_input['affinity_score'].mean():.2f}")
    print(f"\nPor estado:")
    print(df_input['status'].value_counts())
    print(f"\nTop 3 candidatos:")
    print(df_input.nlargest(3, 'affinity_score')[
        ['candidate_id', 'job_id', 'affinity_score', 'status']
    ].to_string(index=False))
    print("="*80)


# ============================================================================
# EJEMPLO 4: ANÁLISIS DE GAPS (BRECHAS DE SKILLS)
# ============================================================================

def ejemplo_analisis_gaps():
    """
    Identificar qué skills le faltan al candidato
    """
    import re
    
    def extraer_skills(texto):
        """Extrae skills mencionadas usando regex simple"""
        # Skills comunes en tech
        skills_conocidas = [
            'Python', 'Java', 'JavaScript', 'React', 'Angular', 'Vue',
            'Django', 'Flask', 'Spring', 'Node.js',
            'PostgreSQL', 'MySQL', 'MongoDB', 'Redis',
            'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP',
            'Git', 'Jenkins', 'CI/CD',
            'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch',
            'HTML', 'CSS', 'SQL'
        ]
        
        texto_lower = texto.lower()
        skills_encontradas = []
        
        for skill in skills_conocidas:
            if skill.lower() in texto_lower:
                skills_encontradas.append(skill)
        
        return set(skills_encontradas)
    
    # Ejemplo
    job = """
    Desarrollador Full Stack
    Skills requeridas: Python, Django, React, PostgreSQL, Docker, AWS, Git
    """
    
    resume = """
    Desarrollador con 3 años de experiencia
    Skills: Python, Django, MySQL, Git, básico Docker
    """
    
    skills_requeridas = extraer_skills(job)
    skills_candidato = extraer_skills(resume)
    
    skills_faltantes = skills_requeridas - skills_candidato
    skills_extra = skills_candidato - skills_requeridas
    skills_match = skills_requeridas & skills_candidato
    
    print("\n" + "="*80)
    print("ANÁLISIS DE GAPS (BRECHAS DE SKILLS)")
    print("="*80)
    
    print(f"\n✅ Skills que coinciden ({len(skills_match)}/{len(skills_requeridas)}):")
    for skill in sorted(skills_match):
        print(f"   • {skill}")
    
    print(f"\n❌ Skills faltantes:")
    for skill in sorted(skills_faltantes):
        print(f"   • {skill}")
    
    print(f"\n➕ Skills adicionales del candidato:")
    for skill in sorted(skills_extra):
        print(f"   • {skill}")
    
    # Calcular porcentaje de match
    match_pct = (len(skills_match) / len(skills_requeridas)) * 100 if skills_requeridas else 0
    
    print(f"\n📊 Porcentaje de match: {match_pct:.1f}%")
    
    print("\n💡 RECOMENDACIÓN:")
    if match_pct >= 80:
        print("   Candidato cumple la mayoría de requisitos. ✅")
    elif match_pct >= 60:
        print("   Candidato tiene buen perfil, pero requiere capacitación en:")
        for skill in sorted(skills_faltantes):
            print(f"      - {skill}")
    else:
        print("   Candidato tiene muchas brechas. Considerar otros perfiles.")
    
    print("="*80)


# ============================================================================
# EJEMPLO 5: INTEGRACIÓN CON API REST
# ============================================================================

def ejemplo_api_integration():
    """
    Ejemplo de cómo crear una API REST para el sistema
    """
    
    # Código de ejemplo para FastAPI
    api_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from job_affinity_model import JobAffinityModel
import pickle

# Inicializar app
app = FastAPI(
    title="Job Affinity API",
    description="API para evaluar afinidad laboral",
    version="1.0.0"
)

# Cargar modelo al inicio
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = JobAffinityModel(max_features=500)
    model.load_model('job_affinity_model.h5')
    
    with open('vectorizers.pkl', 'rb') as f:
        vectorizers = pickle.load(f)
        model.job_vectorizer = vectorizers['job_vectorizer']
        model.resume_vectorizer = vectorizers['resume_vectorizer']
    
    print("✓ Modelo cargado")

# Modelos de datos
class EvaluationRequest(BaseModel):
    job_description: str
    resume: str

class EvaluationResponse(BaseModel):
    affinity_score: float
    interpretation: str
    recommendation: str

# Endpoint principal
@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_candidate(request: EvaluationRequest):
    """
    Evalúa la afinidad entre un trabajo y un candidato
    """
    try:
        # Predecir
        affinity = model.predict_affinity(
            request.job_description,
            request.resume
        )
        
        # Interpretar
        if affinity >= 8.5:
            interpretation = "Excelente"
            recommendation = "Entrevistar con prioridad alta"
        elif affinity >= 7.0:
            interpretation = "Muy Bueno"
            recommendation = "Agendar entrevista"
        elif affinity >= 5.5:
            interpretation = "Bueno"
            recommendation = "Considerar para entrevista"
        elif affinity >= 4.0:
            interpretation = "Aceptable"
            recommendation = "Revisar con detalle"
        else:
            interpretation = "Bajo"
            recommendation = "No cumple requisitos básicos"
        
        return EvaluationResponse(
            affinity_score=round(affinity, 2),
            interpretation=interpretation,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

# Para ejecutar:
# uvicorn api:app --reload --port 8000
    '''
    
    print("\n" + "="*80)
    print("EJEMPLO: API REST CON FASTAPI")
    print("="*80)
    print("\nGuarda este código en 'api.py' y ejecuta:")
    print("   pip install fastapi uvicorn")
    print("   uvicorn api:app --reload --port 8000")
    print("\nLuego prueba con:")
    print("   curl -X POST http://localhost:8000/evaluate \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"job_description\": \"...\", \"resume\": \"...\"}'")
    print("\n" + "="*80)
    print("\nCódigo de la API:\n")
    print(api_code)


# ============================================================================
# MENÚ PRINCIPAL DE EJEMPLOS
# ============================================================================

def main():
    """
    Menú para ejecutar diferentes ejemplos
    """
    print("\n" + "🎯"*40)
    print("EJEMPLOS PRÁCTICOS - SISTEMA DE AFINIDAD LABORAL")
    print("🎯"*40 + "\n")
    
    while True:
        print("\nSelecciona un ejemplo:")
        print("1. Evaluación individual de un candidato")
        print("2. Ranking de múltiples candidatos")
        print("3. Procesamiento en lote desde CSV")
        print("4. Análisis de gaps (brechas de skills)")
        print("5. Ver código de integración con API REST")
        print("6. Salir")
        
        opcion = input("\nOpción (1-6): ").strip()
        
        try:
            if opcion == '1':
                ejemplo_evaluacion_individual()
            elif opcion == '2':
                ejemplo_ranking_candidatos()
            elif opcion == '3':
                ejemplo_batch_processing()
            elif opcion == '4':
                ejemplo_analisis_gaps()
            elif opcion == '5':
                ejemplo_api_integration()
            elif opcion == '6':
                print("\n¡Hasta pronto! 👋\n")
                break
            else:
                print("\n⚠ Opción inválida")
        
        except FileNotFoundError:
            print("\n⚠ Error: Modelo no encontrado.")
            print("Por favor, ejecuta primero:")
            print("   1. python job_affinity_dataset.py")
            print("   2. python job_affinity_model.py")
            break
        
        except Exception as e:
            print(f"\n⚠ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
