"""
Script para usar el modelo de afinidad laboral entrenado
Permite evaluar la afinidad entre trabajos y candidatos de forma interactiva
"""

import numpy as np
from job_affinity_model import JobAffinityModel
import pickle

def load_trained_model():
    """
    Carga el modelo previamente entrenado
    """
    print("Cargando modelo entrenado...")
    model = JobAffinityModel(max_features=500)
    
    try:
        # Cargar el modelo de Keras
        model.load_model('job_affinity_model.h5')
        
        # Cargar vectorizadores (deben guardarse por separado)
        with open('vectorizers.pkl', 'rb') as f:
            vectorizers = pickle.load(f)
            model.job_vectorizer = vectorizers['job_vectorizer']
            model.resume_vectorizer = vectorizers['resume_vectorizer']
        
        print("✓ Modelo cargado exitosamente\n")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Por favor, entrena el modelo primero ejecutando: python job_affinity_model.py")
        return None


def get_affinity_interpretation(score):
    """
    Interpreta el puntaje de afinidad
    """
    if score >= 8.5:
        return "🟢 Excelente - Candidato altamente calificado"
    elif score >= 7.0:
        return "🟢 Muy Bueno - Candidato muy adecuado para el puesto"
    elif score >= 5.5:
        return "🟡 Bueno - Candidato con potencial, considerar entrevista"
    elif score >= 4.0:
        return "🟡 Aceptable - Candidato con algunas carencias"
    elif score >= 2.5:
        return "🔴 Bajo - Candidato poco adecuado"
    else:
        return "🔴 Muy Bajo - Candidato no cumple requisitos básicos"


def evaluate_single_candidate():
    """
    Evalúa un solo candidato para un puesto
    """
    print("="*80)
    print("EVALUACIÓN DE CANDIDATO INDIVIDUAL")
    print("="*80)
    
    # Cargar modelo
    model = load_trained_model()
    if model is None:
        return
    
    # Solicitar descripción del trabajo
    print("\n--- DESCRIPCIÓN DEL TRABAJO ---")
    print("Ingresa la descripción completa del puesto (o presiona Enter para usar ejemplo):")
    job_description = input().strip()
    
    if not job_description:
        job_description = """Puesto: Desarrollador Full Stack. Ubicación: Remoto. 
        Experiencia requerida: 3-5 años. Educación: Profesional en Ingeniería. 
        Skills técnicas requeridas: Python, Django, React, JavaScript, PostgreSQL, AWS, Docker. 
        Skills blandas: Trabajo en equipo, Comunicación efectiva, Resolución de problemas. 
        Idiomas: Inglés avanzado."""
        print("(Usando ejemplo predeterminado)")
    
    # Solicitar CV
    print("\n--- HOJA DE VIDA DEL CANDIDATO ---")
    print("Ingresa la hoja de vida del candidato (o presiona Enter para usar ejemplo):")
    resume = input().strip()
    
    if not resume:
        resume = """Experiencia: 4 años como desarrollador full stack. Educación: Ingeniería de Sistemas. 
        Skills: Python, Django, React, Node.js, PostgreSQL, Git, Docker, AWS básico, 
        Trabajo en equipo, Liderazgo, Gestión del tiempo. 
        Idiomas: Español nativo, Inglés intermedio."""
        print("(Usando ejemplo predeterminado)")
    
    # Predecir afinidad
    print("\nCalculando afinidad...\n")
    affinity = model.predict_affinity(job_description, resume)
    interpretation = get_affinity_interpretation(affinity)
    
    # Mostrar resultados
    print("="*80)
    print("RESULTADO DE LA EVALUACIÓN")
    print("="*80)
    print(f"\n📊 PUNTAJE DE AFINIDAD: {affinity}/10")
    print(f"📝 INTERPRETACIÓN: {interpretation}\n")
    print("="*80)


def evaluate_multiple_candidates():
    """
    Evalúa múltiples candidatos para un mismo puesto
    """
    print("="*80)
    print("EVALUACIÓN DE MÚLTIPLES CANDIDATOS")
    print("="*80)
    
    # Cargar modelo
    model = load_trained_model()
    if model is None:
        return
    
    # Solicitar descripción del trabajo
    print("\n--- DESCRIPCIÓN DEL TRABAJO ---")
    print("Ingresa la descripción completa del puesto:")
    job_description = input().strip()
    
    if not job_description:
        print("⚠ Descripción del trabajo requerida")
        return
    
    # Solicitar número de candidatos
    print("\n¿Cuántos candidatos deseas evaluar?")
    try:
        num_candidates = int(input().strip())
    except:
        print("⚠ Número inválido")
        return
    
    # Evaluar cada candidato
    results = []
    for i in range(num_candidates):
        print(f"\n--- CANDIDATO {i+1} ---")
        print(f"Ingresa la hoja de vida del candidato {i+1}:")
        resume = input().strip()
        
        if resume:
            affinity = model.predict_affinity(job_description, resume)
            results.append({
                'candidato': i+1,
                'affinity': affinity,
                'resume': resume[:100] + "..." if len(resume) > 100 else resume
            })
    
    # Ordenar por afinidad
    results.sort(key=lambda x: x['affinity'], reverse=True)
    
    # Mostrar ranking
    print("\n" + "="*80)
    print("RANKING DE CANDIDATOS")
    print("="*80)
    
    for rank, result in enumerate(results, 1):
        interpretation = get_affinity_interpretation(result['affinity'])
        print(f"\n{rank}. Candidato {result['candidato']}")
        print(f"   Afinidad: {result['affinity']}/10")
        print(f"   {interpretation}")
    
    print("\n" + "="*80)


def batch_evaluation_from_csv():
    """
    Evalúa candidatos desde un archivo CSV
    """
    print("="*80)
    print("EVALUACIÓN EN LOTE DESDE CSV")
    print("="*80)
    
    import pandas as pd
    
    # Cargar modelo
    model = load_trained_model()
    if model is None:
        return
    
    print("\nIngresa la ruta del archivo CSV:")
    print("(El archivo debe tener columnas: 'job_description' y 'resume')")
    csv_path = input().strip()
    
    try:
        df = pd.read_csv(csv_path)
        
        if 'job_description' not in df.columns or 'resume' not in df.columns:
            print("⚠ El CSV debe tener columnas 'job_description' y 'resume'")
            return
        
        print(f"\nEvaluando {len(df)} candidatos...")
        
        # Predecir afinidades
        affinities = []
        for idx, row in df.iterrows():
            affinity = model.predict_affinity(row['job_description'], row['resume'])
            affinities.append(affinity)
        
        df['affinity_score'] = affinities
        df['interpretation'] = df['affinity_score'].apply(get_affinity_interpretation)
        
        # Guardar resultados
        output_path = 'evaluation_results.csv'
        df.to_csv(output_path, index=False)
        
        print(f"\n✓ Evaluación completada")
        print(f"✓ Resultados guardados en: {output_path}")
        
        # Mostrar estadísticas
        print(f"\nEstadísticas:")
        print(f"  Afinidad promedio: {df['affinity_score'].mean():.2f}")
        print(f"  Afinidad máxima: {df['affinity_score'].max():.2f}")
        print(f"  Afinidad mínima: {df['affinity_score'].min():.2f}")
        
    except Exception as e:
        print(f"⚠ Error: {e}")


def show_examples():
    """
    Muestra ejemplos de uso del modelo
    """
    print("="*80)
    print("EJEMPLOS DE USO")
    print("="*80)
    
    # Cargar modelo
    model = load_trained_model()
    if model is None:
        return
    
    examples = [
        {
            'name': 'Desarrollador Backend Senior - Candidato Ideal',
            'job': """Puesto: Desarrollador Backend Senior. Ubicación: Remoto. 
            Experiencia requerida: 5-8 años. Educación: Profesional en Ingeniería. 
            Skills técnicas: Python, Django, PostgreSQL, Redis, Docker, Kubernetes, AWS, Microservicios. 
            Skills blandas: Liderazgo técnico, Mentoría, Trabajo en equipo. 
            Idiomas: Inglés avanzado.""",
            'resume': """Experiencia: 7 años en desarrollo backend. Educación: Ingeniero de Sistemas. 
            Skills: Python, Django, Flask, PostgreSQL, MongoDB, Redis, Docker, Kubernetes, AWS, 
            Arquitectura de microservicios, CI/CD, Liderazgo técnico, Mentoría de juniors. 
            Idiomas: Español nativo, Inglés avanzado, Certificación AWS Solutions Architect."""
        },
        {
            'name': 'Data Scientist - Candidato Junior',
            'job': """Puesto: Data Scientist Senior. Ubicación: Bogotá. 
            Experiencia requerida: 5-8 años. Educación: Maestría en Data Science. 
            Skills técnicas: Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Spark. 
            Skills blandas: Pensamiento analítico, Comunicación de resultados. 
            Idiomas: Inglés avanzado.""",
            'resume': """Experiencia: 1 año como analista de datos. Educación: Profesional en Estadística. 
            Skills: Python básico, SQL, Excel avanzado, PowerBI, Análisis estadístico básico. 
            Idiomas: Español nativo, Inglés intermedio."""
        },
        {
            'name': 'Diseñador UX/UI - Perfil Mixto',
            'job': """Puesto: Diseñador UX/UI. Ubicación: Híbrido. 
            Experiencia requerida: 3-5 años. Educación: Profesional en Diseño. 
            Skills técnicas: Figma, Adobe XD, Sketch, Prototipado, User Research, Design Systems. 
            Skills blandas: Creatividad, Comunicación efectiva, Trabajo colaborativo. 
            Idiomas: Inglés intermedio.""",
            'resume': """Experiencia: 4 años en diseño digital. Educación: Diseñador Gráfico. 
            Skills: Figma, Adobe Photoshop, Illustrator, Prototipado, HTML/CSS básico, 
            Creatividad, Atención al detalle, Trabajo en equipo. 
            Idiomas: Español nativo, Inglés básico."""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- EJEMPLO {i}: {example['name']} ---")
        affinity = model.predict_affinity(example['job'], example['resume'])
        interpretation = get_affinity_interpretation(affinity)
        print(f"📊 Afinidad: {affinity}/10")
        print(f"📝 {interpretation}\n")


def main_menu():
    """
    Menú principal de la aplicación
    """
    while True:
        print("\n" + "="*80)
        print("SISTEMA DE EVALUACIÓN DE AFINIDAD LABORAL")
        print("="*80)
        print("\n¿Qué deseas hacer?")
        print("1. Evaluar un candidato individual")
        print("2. Evaluar múltiples candidatos para un puesto")
        print("3. Evaluación en lote desde CSV")
        print("4. Ver ejemplos de uso")
        print("5. Salir")
        
        choice = input("\nSelecciona una opción (1-5): ").strip()
        
        if choice == '1':
            evaluate_single_candidate()
        elif choice == '2':
            evaluate_multiple_candidates()
        elif choice == '3':
            batch_evaluation_from_csv()
        elif choice == '4':
            show_examples()
        elif choice == '5':
            print("\n¡Hasta pronto!")
            break
        else:
            print("⚠ Opción inválida")


if __name__ == "__main__":
    main_menu()
