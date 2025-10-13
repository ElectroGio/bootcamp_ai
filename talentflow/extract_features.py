"""
Extractor de Features Numéricas para Modelo de Afinidad
Complementa TF-IDF con features estructuradas
"""

import re
import numpy as np

def extract_numeric_features(job_text, resume_text):
    """
    Extrae features numéricas explícitas de los textos
    
    Features extraídas (8 en total):
    1. Ratio de experiencia (candidato/requerido)
    2. Cumple experiencia mínima (binario)
    3. Porcentaje de match de skills técnicas
    4. Número de skills del candidato (normalizado)
    5. Nivel educativo del candidato (normalizado)
    6. Cumple educación requerida (binario)
    7. Nivel de inglés (normalizado)
    8. Es trabajo remoto/híbrido (binario)
    """
    features = []
    
    # 1. AÑOS DE EXPERIENCIA (peso alto en afinidad)
    exp_pattern = r'(\d+)[-]?(\d+)?\s*años?'
    
    job_exp = re.findall(exp_pattern, job_text)
    resume_exp = re.findall(exp_pattern, resume_text)
    
    # Extraer valor numérico (tomar el mínimo del rango)
    job_exp_val = int(job_exp[0][0]) if job_exp else 0
    resume_exp_val = int(resume_exp[0][0]) if resume_exp else 0
    
    # Feature 1: Ratio de experiencia (candidato/requerido)
    exp_ratio = resume_exp_val / max(job_exp_val, 1)
    features.append(min(exp_ratio, 2.0))  # Cap a 2.0 para evitar outliers
    
    # Feature 2: Cumple experiencia (binario)
    features.append(1.0 if resume_exp_val >= job_exp_val else 0.0)
    
    # 2. CONTEO DE SKILLS TÉCNICAS
    tech_skills = [
        'python', 'java', 'javascript', 'react', 'angular', 'vue',
        'django', 'flask', 'spring', 'node', 'sql', 'postgresql',
        'mongodb', 'redis', 'docker', 'kubernetes', 'aws', 'azure',
        'gcp', 'git', 'machine learning', 'deep learning', 
        'tensorflow', 'pytorch', 'scikit', 'c++', 'c#', 'ruby',
        'php', 'swift', 'kotlin', 'go', 'rust', 'typescript',
        'html', 'css', 'sass', 'webpack', 'jenkins', 'ci/cd'
    ]
    
    job_lower = job_text.lower()
    resume_lower = resume_text.lower()
    
    job_skills_count = sum(1 for skill in tech_skills if skill in job_lower)
    resume_skills_count = sum(1 for skill in tech_skills if skill in resume_lower)
    matching_skills = sum(1 for skill in tech_skills 
                          if skill in job_lower and skill in resume_lower)
    
    # Feature 3: Porcentaje de match de skills
    skill_match_pct = matching_skills / max(job_skills_count, 1)
    features.append(skill_match_pct)
    
    # Feature 4: Número de skills del candidato (normalizado)
    features.append(min(resume_skills_count / 10.0, 1.0))
    
    # 3. NIVEL EDUCATIVO
    education_map = {
        'bachiller': 1, 
        'técnico': 2, 
        'tecnólogo': 3,
        'profesional': 4, 
        'maestría': 5, 
        'maestria': 5,
        'doctorado': 6
    }
    
    job_edu = 0
    resume_edu = 0
    
    for edu, level in education_map.items():
        if edu in job_lower:
            job_edu = max(job_edu, level)
        if edu in resume_lower:
            resume_edu = max(resume_edu, level)
    
    # Feature 5: Nivel educativo del candidato (normalizado 0-1)
    features.append(resume_edu / 6.0)
    
    # Feature 6: Cumple educación (binario)
    features.append(1.0 if resume_edu >= job_edu else 0.0)
    
    # 4. NIVEL DE INGLÉS
    english_levels = {
        'básico': 1, 'basic': 1,
        'intermedio': 2, 'intermediate': 2,
        'avanzado': 3, 'advanced': 3,
        'nativo': 4, 'native': 4, 'fluent': 4
    }
    
    resume_english = 0
    for level_text, level_val in english_levels.items():
        if level_text in resume_lower:
            resume_english = max(resume_english, level_val)
    
    # Feature 7: Nivel de inglés (normalizado 0-1)
    features.append(resume_english / 4.0)
    
    # 5. UBICACIÓN Y FLEXIBILIDAD
    locations = ['remoto', 'remote', 'híbrido', 'hybrid', 'anywhere']
    
    # Feature 8: Es remoto/híbrido (más flexible = mejor match)
    is_flexible = any(loc in job_lower for loc in locations)
    features.append(1.0 if is_flexible else 0.0)
    
    return np.array(features, dtype=np.float32)


def get_feature_names():
    """Retorna los nombres de las features para interpretación"""
    return [
        'exp_ratio',
        'meets_exp',
        'skill_match_pct',
        'skill_count_norm',
        'education_level',
        'meets_education',
        'english_level',
        'is_flexible'
    ]


# Test rápido
if __name__ == "__main__":
    job = """
    Desarrollador Senior Python - 5 años experiencia
    Requisitos: Python, Django, PostgreSQL, Docker
    Educación: Profesional en sistemas
    Inglés: Intermedio
    Modalidad: Remoto
    """
    
    resume = """
    Ingeniero de Software con 6 años de experiencia
    Skills: Python, Django, Flask, PostgreSQL, MongoDB, Docker, AWS
    Educación: Profesional
    Inglés: Avanzado
    """
    
    features = extract_numeric_features(job, resume)
    names = get_feature_names()
    
    print("Features extraídas:\n")
    for name, value in zip(names, features):
        print(f"  {name:20s}: {value:.3f}")
    
    print(f"\nTotal features: {len(features)}")
