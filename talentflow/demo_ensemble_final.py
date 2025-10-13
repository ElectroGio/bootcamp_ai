"""
Demo Final - Sistema de Afinidad Laboral con Ensemble
Muestra el uso práctico del modelo entrenado
"""

from ensemble_model import EnsembleAffinityModel

print("="*80)
print("🎯 DEMO: SISTEMA DE EVALUACIÓN DE AFINIDAD LABORAL")
print("="*80)
print("\nCargando modelo ensemble entrenado...")

# Cargar modelo (ya entrenado)
ensemble = EnsembleAffinityModel(load_models=True)

print("\n")
print("="*80)
print("CASOS DE PRUEBA")
print("="*80)

# CASO 1: Alta Afinidad Esperada
print("\n📌 CASO 1: Candidato con Excelente Match")
print("-" * 80)

job1 = """
Desarrollador Full Stack Senior
Ubicación: Remoto
Experiencia requerida: 5-8 años
Educación: Profesional en Ingeniería de Sistemas
Skills técnicas: Python, Django, React, PostgreSQL, Docker, AWS, Git
Skills blandas: Liderazgo, Comunicación efectiva, Trabajo en equipo
Idiomas: Inglés avanzado
"""

resume1 = """
Experiencia: 8+ años
Educación: Profesional en Ingeniería
Skills: Python, Django, Flask, React, JavaScript, PostgreSQL, MongoDB, 
Docker, Kubernetes, AWS, Git, Trabajo en equipo, Liderazgo
Idiomas: Inglés avanzado, Español nativo
"""

print("\n📋 OFERTA LABORAL:")
print(job1.strip())
print("\n👤 CANDIDATO:")
print(resume1.strip())

score1 = ensemble.predict(job1, resume1)
print(f"\n🎯 AFINIDAD PREDICHA: {score1}/10")

if score1 >= 8.0:
    print("✅ RECOMENDACIÓN: EXCELENTE MATCH - Priorizar para entrevista")
elif score1 >= 6.0:
    print("✅ RECOMENDACIÓN: BUEN MATCH - Considerar para entrevista")
elif score1 >= 4.0:
    print("⚠️ RECOMENDACIÓN: MATCH REGULAR - Evaluar más a fondo")
else:
    print("❌ RECOMENDACIÓN: MATCH BAJO - No recomendado")

# CASO 2: Afinidad Media
print("\n" + "="*80)
print("📌 CASO 2: Candidato con Match Parcial")
print("-" * 80)

job2 = """
Data Scientist Senior
Ubicación: Bogotá
Experiencia requerida: 5-8 años
Educación: Maestría
Skills técnicas: Python, Machine Learning, TensorFlow, SQL, AWS
Skills blandas: Pensamiento crítico, Resolución de problemas
Idiomas: Inglés avanzado
"""

resume2 = """
Experiencia: 3-5 años
Educación: Profesional en Ingeniería
Skills: Python, Pandas, Scikit-learn, SQL, Machine Learning, 
Comunicación efectiva, Trabajo en equipo
Idiomas: Inglés intermedio, Español nativo
"""

print("\n📋 OFERTA LABORAL:")
print(job2.strip())
print("\n👤 CANDIDATO:")
print(resume2.strip())

score2 = ensemble.predict(job2, resume2)
print(f"\n🎯 AFINIDAD PREDICHA: {score2}/10")

if score2 >= 8.0:
    print("✅ RECOMENDACIÓN: EXCELENTE MATCH - Priorizar para entrevista")
elif score2 >= 6.0:
    print("✅ RECOMENDACIÓN: BUEN MATCH - Considerar para entrevista")
elif score2 >= 4.0:
    print("⚠️ RECOMENDACIÓN: MATCH REGULAR - Evaluar más a fondo")
else:
    print("❌ RECOMENDACIÓN: MATCH BAJO - No recomendado")

# CASO 3: Baja Afinidad
print("\n" + "="*80)
print("📌 CASO 3: Candidato con Bajo Match")
print("-" * 80)

job3 = """
Gerente de Proyectos Senior
Ubicación: Medellín
Experiencia requerida: 8+ años
Educación: Maestría
Skills técnicas: Gestión de proyectos, Scrum, Agile, PowerBI
Skills blandas: Liderazgo, Negociación, Comunicación efectiva
Idiomas: Inglés nativo
"""

resume3 = """
Experiencia: 1-3 años
Educación: Técnico
Skills: Excel avanzado, PowerBI, Marketing digital, SEO, 
Content creation, Creatividad
Idiomas: Inglés básico, Español nativo
"""

print("\n📋 OFERTA LABORAL:")
print(job3.strip())
print("\n👤 CANDIDATO:")
print(resume3.strip())

score3 = ensemble.predict(job3, resume3)
print(f"\n🎯 AFINIDAD PREDICHA: {score3}/10")

if score3 >= 8.0:
    print("✅ RECOMENDACIÓN: EXCELENTE MATCH - Priorizar para entrevista")
elif score3 >= 6.0:
    print("✅ RECOMENDACIÓN: BUEN MATCH - Considerar para entrevista")
elif score3 >= 4.0:
    print("⚠️ RECOMENDACIÓN: MATCH REGULAR - Evaluar más a fondo")
else:
    print("❌ RECOMENDACIÓN: MATCH BAJO - No recomendado")

# RESUMEN
print("\n" + "="*80)
print("📊 RESUMEN DE EVALUACIONES")
print("="*80)
print(f"\nCaso 1 (Excelente candidato): {score1}/10")
print(f"Caso 2 (Candidato promedio):  {score2}/10")
print(f"Caso 3 (Bajo match):          {score3}/10")

print("\n" + "="*80)
print("✅ DEMO COMPLETADO")
print("="*80)
print("\n💡 PRÓXIMOS PASOS:")
print("   1. Usar ensemble.predict(job, resume) para evaluar candidatos")
print("   2. Integrar con tu sistema de reclutamiento")
print("   3. Revisar documentación en GUIA_USO_FINAL.md")
print("\n📚 Archivos importantes:")
print("   - ensemble_model.py: Modelo principal")
print("   - GUIA_USO_FINAL.md: Manual completo")
print("   - INFORME_EJECUTIVO.md: Resumen ejecutivo")
