"""
Script para regenerar el dataset con las mejoras de precisión
Ejecuta este script para crear un nuevo dataset optimizado
"""

import os
import sys

def main():
    print("="*80)
    print("REGENERACIÓN DE DATASET CON MEJORAS DE PRECISIÓN")
    print("="*80)
    print("\n📊 MEJORAS IMPLEMENTADAS:")
    print("   ✅ Factor aleatorio reducido: (0, 3.0) → (0, 1.5)")
    print("   ✅ Peso de skills aumentado: 40% → 50%")
    print("   ✅ Peso de experiencia aumentado: 30% → 35%")
    print("   ✅ Cálculo de experiencia más granular")
    print("   ✅ Tamaño del dataset: 2000 → 5000 muestras")
    print("\n" + "="*80)
    
    # Verificar si existe dataset anterior
    if os.path.exists('job_affinity_dataset.csv'):
        print("\n⚠ Ya existe un dataset anterior: job_affinity_dataset.csv")
        respuesta = input("¿Deseas crear un backup y regenerar? (s/n): ").strip().lower()
        
        if respuesta == 's':
            # Hacer backup
            import shutil
            backup_name = 'job_affinity_dataset_backup.csv'
            shutil.copy('job_affinity_dataset.csv', backup_name)
            print(f"✓ Backup guardado en: {backup_name}")
        else:
            print("\n❌ Operación cancelada")
            return
    
    # Regenerar dataset
    print("\n🔄 Regenerando dataset...")
    print("   (Esto tomará un momento debido al mayor tamaño: 5000 muestras)\n")
    
    import subprocess
    result = subprocess.run(['python', 'job_affinity_dataset.py'], 
                          capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("✅ DATASET MEJORADO GENERADO EXITOSAMENTE")
        print("="*80)
        print("\n📈 PRÓXIMOS PASOS:")
        print("   1. Entrena el modelo con el nuevo dataset:")
        print("      python job_affinity_model.py")
        print("\n   2. Resultados esperados después del entrenamiento:")
        print("      • MAE < 0.7 (mejora significativa)")
        print("      • R² Score > 0.85")
        print("      • 85%+ predicciones con error ≤ 1.0")
        print("      • Menor dispersión en las predicciones")
        print("\n" + "="*80)
    else:
        print("\n❌ Error al generar el dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()
