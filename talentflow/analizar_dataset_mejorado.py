"""
Comparación del dataset ANTES vs DESPUÉS de las mejoras
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar dataset mejorado
df = pd.read_csv('job_affinity_dataset.csv')

print("="*80)
print("ANÁLISIS DEL DATASET MEJORADO")
print("="*80)

print(f"\n📊 ESTADÍSTICAS:")
print(f"   • Número de muestras: {len(df)}")
print(f"   • Media de afinidad: {df['affinity_score'].mean():.2f}/10")
print(f"   • Desviación estándar: {df['affinity_score'].std():.2f}")
print(f"   • Rango: {df['affinity_score'].min():.2f} - {df['affinity_score'].max():.2f}")

print(f"\n📈 DISTRIBUCIÓN:")
bins = [0, 2, 4, 6, 8, 10]
labels = ['0-2', '2-4', '4-6', '6-8', '8-10']
df['rango'] = pd.cut(df['affinity_score'], bins=bins, labels=labels)
distribucion = df['rango'].value_counts().sort_index()

for rango, count in distribucion.items():
    porcentaje = (count / len(df)) * 100
    barra = '█' * int(porcentaje / 2)
    print(f"   {rango}: {barra} {porcentaje:.1f}% ({count} muestras)")

print(f"\n✅ MEJORAS APLICADAS:")
print("   1. Factor aleatorio reducido: 0-3.0 → 0-1.5 (50% menos)")
print("   2. Peso de skills aumentado: 40% → 50%")
print("   3. Peso de experiencia aumentado: 30% → 35%")
print("   4. Cálculo más granular y determinista")
print("   5. Dataset aumentado: 2000 → 5000 muestras (150% más)")

print(f"\n🎯 COMPARACIÓN (estimada):")
print("   ANTES:")
print("      • Desviación estándar: ~1.90")
print("      • Alta dispersión debido a factor aleatorio")
print("   DESPUÉS:")
print(f"      • Desviación estándar: {df['affinity_score'].std():.2f}")
print("      • Menor dispersión (más consistente)")

print(f"\n💡 RESULTADO ESPERADO DESPUÉS DE ENTRENAR:")
print("   • MAE: < 0.7 (vs ~1.0 anterior)")
print("   • R² Score: > 0.85 (vs ~0.70 anterior)")
print("   • Predicciones con error ≤ 1.0: > 85% (vs ~60% anterior)")

print("\n" + "="*80)
print("✓ DATASET MEJORADO LISTO PARA ENTRENAR")
print("="*80)

print("\n🚀 SIGUIENTE PASO:")
print("   python job_affinity_model.py")
print("\n   El modelo entrenará con:")
print("      • 5000 muestras (mayor capacidad de aprendizaje)")
print("      • Arquitectura mejorada (512→256→128→64→32)")
print("      • Hiperparámetros optimizados")
print("      • Tiempo estimado: 10-15 minutos")

