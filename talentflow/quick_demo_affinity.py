"""
Script de ejemplo rápido para probar el sistema de afinidad laboral
Ejecuta todo el pipeline: genera datos, entrena modelo y hace predicciones
"""

import os
import sys

def quick_demo():
    """
    Demo rápido del sistema completo
    """
    print("="*80)
    print("DEMO RÁPIDO - SISTEMA DE AFINIDAD LABORAL")
    print("="*80)
    
    # Paso 1: Generar dataset
    print("\n[1/3] Generando dataset sintético...")
    print("-" * 80)
    
    if not os.path.exists('job_affinity_dataset.csv'):
        print("Ejecutando generador de dataset...")
        os.system('python job_affinity_dataset.py')
    else:
        print("✓ Dataset ya existe: job_affinity_dataset.csv")
    
    # Paso 2: Entrenar modelo (versión reducida)
    print("\n[2/3] Entrenando modelo...")
    print("-" * 80)
    
    if not os.path.exists('job_affinity_model.h5'):
        print("Ejecutando entrenamiento del modelo...")
        print("(Esto puede tomar varios minutos)")
        os.system('python job_affinity_model.py')
    else:
        print("✓ Modelo ya entrenado: job_affinity_model.h5")
        respuesta = input("\n¿Deseas re-entrenar el modelo? (s/n): ").strip().lower()
        if respuesta == 's':
            os.system('python job_affinity_model.py')
    
    # Paso 3: Hacer predicciones de ejemplo
    print("\n[3/3] Probando predicciones...")
    print("-" * 80)
    
    # Importar el modelo
    from job_affinity_model import JobAffinityModel
    import pickle
    
    # Cargar modelo
    print("\nCargando modelo entrenado...")
    model = JobAffinityModel(max_features=500)
    
    try:
        model.load_model('job_affinity_model.h5')
        
        with open('vectorizers.pkl', 'rb') as f:
            vectorizers = pickle.load(f)
            model.job_vectorizer = vectorizers['job_vectorizer']
            model.resume_vectorizer = vectorizers['resume_vectorizer']
        
        print("✓ Modelo cargado exitosamente\n")
        
        # Ejemplos de predicción
        ejemplos = [
            {
                'titulo': 'Desarrollador Full Stack Senior - Match Alto',
                'job': """Puesto: Desarrollador Full Stack Senior. Ubicación: Remoto. 
                Experiencia requerida: 5-8 años. Educación: Profesional en Ingeniería. 
                Skills técnicas requeridas: Python, Django, React, PostgreSQL, Docker, AWS. 
                Skills blandas: Liderazgo, Trabajo en equipo, Comunicación efectiva. 
                Idiomas: Inglés avanzado.""",
                'resume': """Experiencia: 7 años como desarrollador full stack en empresas tech. 
                Educación: Ingeniero de Sistemas, Maestría en Software Engineering. 
                Skills: Python, Django, Flask, React, Vue.js, Node.js, PostgreSQL, MongoDB, 
                Docker, Kubernetes, AWS, CI/CD, Git, Liderazgo técnico, Mentoría, Scrum. 
                Idiomas: Español nativo, Inglés avanzado (C1), Certificaciones AWS.""",
                'esperado': 'Alto (8-10)'
            },
            {
                'titulo': 'Data Scientist - Match Medio',
                'job': """Puesto: Data Scientist. Ubicación: Bogotá. 
                Experiencia requerida: 3-5 años. Educación: Maestría en Data Science. 
                Skills técnicas requeridas: Python, Machine Learning, TensorFlow, SQL, Pandas. 
                Skills blandas: Pensamiento analítico, Resolución de problemas. 
                Idiomas: Inglés avanzado.""",
                'resume': """Experiencia: 3 años como analista de datos. 
                Educación: Profesional en Estadística, Diplomado en Machine Learning. 
                Skills: Python, SQL, Pandas, Scikit-learn, PowerBI, Análisis estadístico, 
                Visualización de datos, Excel avanzado, Trabajo en equipo. 
                Idiomas: Español nativo, Inglés intermedio.""",
                'esperado': 'Medio (5-7)'
            },
            {
                'titulo': 'Machine Learning Engineer - Match Bajo',
                'job': """Puesto: Machine Learning Engineer. Ubicación: Remoto. 
                Experiencia requerida: 5-8 años. Educación: Maestría o Doctorado. 
                Skills técnicas requeridas: Python, Deep Learning, TensorFlow, PyTorch, 
                MLOps, Kubernetes, Cloud Computing. 
                Skills blandas: Investigación, Innovación. Idiomas: Inglés nativo.""",
                'resume': """Experiencia: 1 año como desarrollador junior. 
                Educación: Técnico en programación. 
                Skills: Python básico, HTML, CSS, JavaScript, MySQL básico. 
                Idiomas: Español nativo, Inglés básico.""",
                'esperado': 'Bajo (0-4)'
            }
        ]
        
        print("="*80)
        print("EJEMPLOS DE PREDICCIÓN")
        print("="*80)
        
        for i, ejemplo in enumerate(ejemplos, 1):
            print(f"\n📋 EJEMPLO {i}: {ejemplo['titulo']}")
            print(f"   Afinidad esperada: {ejemplo['esperado']}")
            print("-" * 80)
            
            # Predecir
            afinidad = model.predict_affinity(ejemplo['job'], ejemplo['resume'])
            
            # Interpretación
            if afinidad >= 8.5:
                interpretacion = "🟢 Excelente - Candidato altamente calificado"
            elif afinidad >= 7.0:
                interpretacion = "🟢 Muy Bueno - Candidato muy adecuado"
            elif afinidad >= 5.5:
                interpretacion = "🟡 Bueno - Candidato con potencial"
            elif afinidad >= 4.0:
                interpretacion = "🟡 Aceptable - Algunas carencias"
            elif afinidad >= 2.5:
                interpretacion = "🔴 Bajo - Poco adecuado"
            else:
                interpretacion = "🔴 Muy Bajo - No cumple requisitos"
            
            print(f"\n   📊 AFINIDAD PREDICHA: {afinidad}/10")
            print(f"   📝 {interpretacion}\n")
        
        print("="*80)
        print("✓ DEMO COMPLETADO EXITOSAMENTE")
        print("="*80)
        
        print("\n💡 PRÓXIMOS PASOS:")
        print("   1. Ejecuta 'python job_affinity_predictor.py' para usar el menú interactivo")
        print("   2. Lee 'README_AFINIDAD.md' para más información")
        print("   3. Personaliza los ejemplos en 'job_affinity_dataset.py'")
        
    except Exception as e:
        print(f"\n⚠ Error al cargar el modelo: {e}")
        print("Por favor, ejecuta 'python job_affinity_model.py' primero")


def verificar_dependencias():
    """
    Verifica que todas las dependencias estén instaladas
    """
    print("Verificando dependencias...")
    
    dependencias = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'matplotlib': 'matplotlib',
        'nltk': 'nltk'
    }
    
    faltantes = []
    
    for modulo, paquete in dependencias.items():
        try:
            __import__(modulo)
            print(f"  ✓ {paquete}")
        except ImportError:
            print(f"  ✗ {paquete} NO INSTALADO")
            faltantes.append(paquete)
    
    if faltantes:
        print(f"\n⚠ Faltan dependencias. Instala con:")
        print(f"   pip install {' '.join(faltantes)}")
        return False
    
    print("\n✓ Todas las dependencias están instaladas\n")
    return True


if __name__ == "__main__":
    print("\n" + "🚀" * 40)
    print("SISTEMA DE EVALUACIÓN DE AFINIDAD LABORAL - DEMO RÁPIDO")
    print("🚀" * 40 + "\n")
    
    # Verificar dependencias
    if not verificar_dependencias():
        sys.exit(1)
    
    # Ejecutar demo
    try:
        quick_demo()
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n⚠ Error durante el demo: {e}")
        import traceback
        traceback.print_exc()
