"""
Generador de Dataset Sintético para Evaluación de Afinidad Laboral
Este script genera un dataset con descripciones de trabajos, hojas de vida y puntajes de afinidad
"""

import pandas as pd
import numpy as np
import random

# Configurar semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

# Definir catálogos de datos
SKILLS = {
    'tech': ['Python', 'Java', 'JavaScript', 'C++', 'SQL', 'React', 'Node.js', 'Django', 
             'Machine Learning', 'Deep Learning', 'TensorFlow', 'Keras', 'AWS', 'Docker', 
             'Kubernetes', 'Git', 'APIs REST', 'MongoDB', 'PostgreSQL', 'Azure'],
    'business': ['Análisis financiero', 'Excel avanzado', 'PowerBI', 'Tableau', 'SAP',
                 'Gestión de proyectos', 'Scrum', 'Agile', 'Liderazgo', 'Negociación'],
    'creative': ['Diseño gráfico', 'Adobe Photoshop', 'Illustrator', 'Figma', 'UX/UI',
                 'Branding', 'Marketing digital', 'SEO', 'Content creation', 'Video editing'],
    'soft': ['Trabajo en equipo', 'Comunicación efectiva', 'Resolución de problemas',
             'Pensamiento crítico', 'Adaptabilidad', 'Creatividad', 'Gestión del tiempo']
}

POSITIONS = [
    'Desarrollador Full Stack', 'Data Scientist', 'Ingeniero de Machine Learning',
    'Diseñador UX/UI', 'Analista de Datos', 'Gerente de Proyectos',
    'Desarrollador Backend', 'Desarrollador Frontend', 'DevOps Engineer',
    'Analista Financiero', 'Especialista en Marketing Digital', 'Product Manager'
]

EDUCATION_LEVELS = [
    'Bachiller', 'Técnico', 'Tecnólogo', 'Profesional en Ingeniería',
    'Profesional en Administración', 'Profesional en Diseño', 'Maestría', 'Doctorado'
]

LOCATIONS = ['Remoto', 'Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Híbrido']

EXPERIENCE_RANGES = ['0-1 años', '1-3 años', '3-5 años', '5-8 años', '8+ años']

LANGUAGES = ['Español nativo', 'Inglés básico', 'Inglés intermedio', 'Inglés avanzado', 
             'Inglés nativo', 'Portugués', 'Francés']


def generate_job_description(position_type='tech'):
    """Genera una descripción de trabajo realista"""
    position = random.choice(POSITIONS)
    
    # Seleccionar skills relevantes
    num_tech_skills = random.randint(3, 7)
    num_soft_skills = random.randint(2, 4)
    
    tech_skills = random.sample(SKILLS['tech'], min(num_tech_skills, len(SKILLS['tech'])))
    soft_skills = random.sample(SKILLS['soft'], min(num_soft_skills, len(SKILLS['soft'])))
    
    experience = random.choice(EXPERIENCE_RANGES)
    education = random.choice(EDUCATION_LEVELS[3:])  # Mínimo profesional
    location = random.choice(LOCATIONS)
    language = random.choice(LANGUAGES[2:])  # Mínimo inglés intermedio para tech
    
    description = f"Puesto: {position}. "
    description += f"Ubicación: {location}. "
    description += f"Experiencia requerida: {experience}. "
    description += f"Educación: {education}. "
    description += f"Skills técnicas requeridas: {', '.join(tech_skills)}. "
    description += f"Skills blandas: {', '.join(soft_skills)}. "
    description += f"Idiomas: {language}."
    
    return {
        'job_description': description,
        'position': position,
        'skills': tech_skills + soft_skills,
        'experience': experience,
        'location': location
    }


def generate_resume(match_job=None, match_level='high'):
    """
    Genera una hoja de vida realista
    match_job: diccionario con información del trabajo para hacer match
    match_level: 'high', 'medium', 'low', 'random' - nivel de coincidencia con el trabajo
    """
    
    if match_job and match_level != 'random':
        # Generar CV con cierto nivel de coincidencia
        job_skills = match_job['skills']
        
        if match_level == 'high':
            # 70-90% de coincidencia en skills
            match_ratio = random.uniform(0.7, 0.9)
            num_matching_skills = int(len(job_skills) * match_ratio)
            candidate_skills = random.sample(job_skills, num_matching_skills)
            # Agregar algunas skills adicionales
            additional_skills = random.sample([s for s in SKILLS['tech'] + SKILLS['soft'] 
                                              if s not in candidate_skills], random.randint(2, 5))
            candidate_skills.extend(additional_skills)
            
            # Experiencia similar o mayor
            exp_levels = EXPERIENCE_RANGES
            job_exp_idx = exp_levels.index(match_job['experience'])
            candidate_exp = random.choice(exp_levels[max(0, job_exp_idx-1):])
            
        elif match_level == 'medium':
            # 40-60% de coincidencia
            match_ratio = random.uniform(0.4, 0.6)
            num_matching_skills = int(len(job_skills) * match_ratio)
            candidate_skills = random.sample(job_skills, num_matching_skills)
            # Más skills no relacionadas
            additional_skills = random.sample([s for s in SKILLS['tech'] + SKILLS['business'] 
                                              if s not in candidate_skills], random.randint(3, 7))
            candidate_skills.extend(additional_skills)
            
            # Experiencia puede ser menor
            exp_levels = EXPERIENCE_RANGES
            candidate_exp = random.choice(exp_levels)
            
        else:  # low
            # 10-30% de coincidencia
            match_ratio = random.uniform(0.1, 0.3)
            num_matching_skills = max(1, int(len(job_skills) * match_ratio))
            candidate_skills = random.sample(job_skills, num_matching_skills)
            # Muchas skills no relacionadas
            additional_skills = random.sample([s for s in SKILLS['creative'] + SKILLS['business'] 
                                              if s not in candidate_skills], random.randint(4, 8))
            candidate_skills.extend(additional_skills)
            
            # Experiencia menor
            candidate_exp = random.choice(EXPERIENCE_RANGES[:3])
    else:
        # Generar CV aleatorio
        num_skills = random.randint(4, 10)
        all_skills = SKILLS['tech'] + SKILLS['business'] + SKILLS['creative'] + SKILLS['soft']
        candidate_skills = random.sample(all_skills, num_skills)
        candidate_exp = random.choice(EXPERIENCE_RANGES)
    
    education = random.choice(EDUCATION_LEVELS)
    languages = random.sample(LANGUAGES, random.randint(1, 3))
    
    resume = f"Experiencia: {candidate_exp}. "
    resume += f"Educación: {education}. "
    resume += f"Skills: {', '.join(candidate_skills)}. "
    resume += f"Idiomas: {', '.join(languages)}."
    
    return {
        'resume': resume,
        'skills': candidate_skills,
        'experience': candidate_exp,
        'education': education
    }


def calculate_affinity_score(job, resume):
    """
    Calcula el puntaje de afinidad basado en coincidencias
    Retorna un valor entre 0 y 10
    """
    score = 0.0
    
    # 1. Coincidencia de skills (50% del puntaje) - MEJORADO
    job_skills_set = set(job['skills'])
    resume_skills_set = set(resume['skills'])
    
    if len(job_skills_set) > 0:
        skills_match = len(job_skills_set.intersection(resume_skills_set)) / len(job_skills_set)
        score += skills_match * 5.0  # Aumentado de 4.0 a 5.0 para mayor peso en skills
    
    # 2. Experiencia (35% del puntaje) - MEJORADO
    exp_levels = EXPERIENCE_RANGES
    try:
        job_exp_idx = exp_levels.index(job['experience'])
        resume_exp_idx = exp_levels.index(resume['experience'])
        
        # Cálculo más granular y realista
        if resume_exp_idx >= job_exp_idx:
            score += 3.5  # Cumple o supera el requisito
        elif resume_exp_idx == job_exp_idx - 1:
            score += 2.5  # 1 nivel por debajo - aún aceptable
        elif resume_exp_idx == job_exp_idx - 2:
            score += 1.0  # 2 niveles por debajo - deficiente
        else:
            score += 0.2  # Muy por debajo del requisito
    except:
        score += 1.5
    
    # 3. Factor de variabilidad ULTRA REDUCIDO (5% del puntaje) 
    # Simula factores como cultura, soft skills, motivación, etc.
    # MEJORADO v3: Reducido de (0, 0.8) a (0, 0.5) para máxima precisión
    score += random.uniform(0, 0.5)
    
    # Asegurar que esté entre 0 y 10
    score = max(0.0, min(10.0, score))
    
    return round(score, 2)


def generate_dataset(num_samples=5000):
    """
    Genera el dataset completo
    MEJORADO: Aumentado de 1000 a 5000 muestras para mejor aprendizaje
    """
    data = []
    
    # Generar muestras con diferentes niveles de coincidencia
    for i in range(num_samples):
        # Determinar nivel de match para esta muestra
        rand = random.random()
        if rand < 0.3:
            match_level = 'high'
        elif rand < 0.6:
            match_level = 'medium'
        elif rand < 0.85:
            match_level = 'low'
        else:
            match_level = 'random'
        
        # Generar trabajo y CV
        job = generate_job_description()
        resume = generate_resume(match_job=job, match_level=match_level)
        
        # Calcular afinidad
        affinity = calculate_affinity_score(job, resume)
        
        data.append({
            'job_description': job['job_description'],
            'resume': resume['resume'],
            'affinity_score': affinity
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Generando dataset de afinidad laboral...")
    print("VERSIÓN ULTRA OPTIMIZADA: Máxima precisión y mínima dispersión\n")
    
    # Generar dataset - OPTIMIZADO: 10000 muestras
    df = generate_dataset(num_samples=10000)
    
    # Mostrar estadísticas
    print(f"\nDataset generado con {len(df)} muestras")
    print(f"\nEstadísticas de afinidad:")
    print(df['affinity_score'].describe())
    
    print(f"\nDistribución de puntajes:")
    print(df['affinity_score'].value_counts(bins=5).sort_index())
    
    # Guardar dataset
    output_path = 'job_affinity_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Dataset guardado en: {output_path}")
    
    # Mostrar ejemplos
    print("\n" + "="*80)
    print("EJEMPLOS DEL DATASET:")
    print("="*80)
    
    for idx in [0, len(df)//2, len(df)-1]:
        print(f"\n--- Ejemplo {idx + 1} ---")
        print(f"TRABAJO:\n{df.iloc[idx]['job_description']}\n")
        print(f"HOJA DE VIDA:\n{df.iloc[idx]['resume']}\n")
        print(f"AFINIDAD: {df.iloc[idx]['affinity_score']}/10")
        print("-" * 80)
