"""
Modelo Ensemble para Afinidad Laboral
Combina: Red Neuronal + Random Forest + Gradient Boosting
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# Importar el modelo base
from job_affinity_model import JobAffinityModel

class EnsembleAffinityModel:
    def __init__(self, load_models=False):
        self.nn_model = JobAffinityModel()
        self.rf_model = None
        self.gb_model = None
        self.weights = {'nn': 0.5, 'rf': 0.25, 'gb': 0.25}
        
        # Cargar modelos si existen
        if load_models:
            self.load()
        
    def train(self, dataset_path='job_affinity_dataset.csv'):
        """Entrena los tres modelos"""
        print("="*80)
        print("MODELO ENSEMBLE DE AFINIDAD LABORAL")
        print("="*80)
        
        # 1. Entrenar Red Neuronal
        print("\n[1/3] Entrenando Red Neuronal...")
        X_job_train, X_job_test, X_resume_train, X_resume_test, y_train, y_test = \
            self.nn_model.load_and_prepare_data(dataset_path, test_size=0.2)
        
        X_train, X_test = self.nn_model.vectorize_text(
            X_job_train, X_job_test, X_resume_train, X_resume_test
        )
        
        self.nn_model.build_model(X_train.shape[1])
        self.nn_model.train(X_train, y_train, X_test, y_test)
        
        # 2. Entrenar Random Forest
        print("\n[2/3] Entrenando Random Forest...")
        self.rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        self.rf_model.fit(X_train, y_train)
        
        # 3. Entrenar Gradient Boosting
        print("\n[3/3] Entrenando Gradient Boosting...")
        self.gb_model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            verbose=1
        )
        self.gb_model.fit(X_train, y_train)
        
        # 4. Evaluar ensemble
        print("\n" + "="*80)
        print("EVALUACIÓN DEL ENSEMBLE")
        print("="*80)
        
        # Predicciones individuales
        pred_nn = self.nn_model.model.predict(X_test, verbose=0).flatten()
        pred_rf = self.rf_model.predict(X_test)
        pred_gb = self.gb_model.predict(X_test)
        
        # Predicción ensemble (promedio ponderado)
        pred_ensemble = (
            self.weights['nn'] * pred_nn + 
            self.weights['rf'] * pred_rf + 
            self.weights['gb'] * pred_gb
        )
        pred_ensemble = np.clip(pred_ensemble, 0, 10)
        
        # Métricas individuales
        print("\n📊 Métricas por modelo:")
        print("\nRed Neuronal:")
        print(f"  MAE: {mean_absolute_error(y_test, pred_nn):.4f}")
        print(f"  R²:  {r2_score(y_test, pred_nn):.4f}")
        
        print("\nRandom Forest:")
        print(f"  MAE: {mean_absolute_error(y_test, pred_rf):.4f}")
        print(f"  R²:  {r2_score(y_test, pred_rf):.4f}")
        
        print("\nGradient Boosting:")
        print(f"  MAE: {mean_absolute_error(y_test, pred_gb):.4f}")
        print(f"  R²:  {r2_score(y_test, pred_gb):.4f}")
        
        # Métricas ensemble
        mae_ensemble = mean_absolute_error(y_test, pred_ensemble)
        mse_ensemble = mean_squared_error(y_test, pred_ensemble)
        rmse_ensemble = np.sqrt(mse_ensemble)
        r2_ensemble = r2_score(y_test, pred_ensemble)
        
        print("\n" + "="*80)
        print("🏆 ENSEMBLE (Combinado):")
        print(f"  MAE:  {mae_ensemble:.4f}")
        print(f"  RMSE: {rmse_ensemble:.4f}")
        print(f"  R²:   {r2_ensemble:.4f}")
        print("="*80)
        
        # Análisis de errores
        errors = np.abs(y_test - pred_ensemble)
        print(f"\nAnálisis de errores:")
        print(f"  Error mínimo: {errors.min():.4f}")
        print(f"  Error máximo: {errors.max():.4f}")
        print(f"  Error mediano: {np.median(errors):.4f}")
        
        # Precisión por rangos
        error_05 = np.sum(errors <= 0.5) / len(errors) * 100
        error_10 = np.sum(errors <= 1.0) / len(errors) * 100
        error_15 = np.sum(errors <= 1.5) / len(errors) * 100
        
        print(f"\n🎯 Precisión por rango:")
        print(f"  Predicciones con error ≤ 0.5: {error_05:.2f}%")
        print(f"  Predicciones con error ≤ 1.0: {error_10:.2f}%")
        print(f"  Predicciones con error ≤ 1.5: {error_15:.2f}%")
        
        # Visualización comparativa
        self._plot_comparison(y_test, pred_nn, pred_rf, pred_gb, pred_ensemble)
        
        return X_test, y_test
    
    def _plot_comparison(self, y_true, pred_nn, pred_rf, pred_gb, pred_ensemble):
        """Genera gráfico comparativo de modelos"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        models = [
            ('Red Neuronal', pred_nn, axes[0, 0]),
            ('Random Forest', pred_rf, axes[0, 1]),
            ('Gradient Boosting', pred_gb, axes[1, 0]),
            ('ENSEMBLE', pred_ensemble, axes[1, 1])
        ]
        
        for title, pred, ax in models:
            ax.scatter(y_true, pred, alpha=0.5, s=20)
            ax.plot([0, 10], [0, 10], 'r--', linewidth=2, label='Predicción perfecta')
            
            mae = mean_absolute_error(y_true, pred)
            r2 = r2_score(y_true, pred)
            
            ax.set_xlabel('Afinidad Real', fontsize=11)
            ax.set_ylabel('Afinidad Predicha', fontsize=11)
            ax.set_title(f'{title}\nMAE={mae:.3f}, R²={r2:.3f}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
        
        plt.tight_layout()
        plt.savefig('ensemble_comparison.png', dpi=150, bbox_inches='tight')
        print("\n✓ Gráfico comparativo guardado: ensemble_comparison.png")
    
    def predict(self, job_description, resume):
        """Predice usando ensemble"""
        # Predicción NN
        pred_nn = self.nn_model.predict_affinity(job_description, resume)
        
        # Preparar para RF y GB
        job_clean = self.nn_model.preprocess_text(job_description)
        resume_clean = self.nn_model.preprocess_text(resume)
        
        job_vec = self.nn_model.job_vectorizer.transform([job_clean]).toarray()
        resume_vec = self.nn_model.resume_vectorizer.transform([resume_clean]).toarray()
        
        from extract_features import extract_numeric_features
        numeric_feats = extract_numeric_features(job_description, resume).reshape(1, -1)
        
        X_combined = np.concatenate([job_vec, resume_vec, numeric_feats], axis=1)
        
        # Predicción RF y GB
        pred_rf = self.rf_model.predict(X_combined)[0]
        pred_gb = self.gb_model.predict(X_combined)[0]
        
        # Ensemble
        pred_ensemble = (
            self.weights['nn'] * pred_nn + 
            self.weights['rf'] * pred_rf + 
            self.weights['gb'] * pred_gb
        )
        pred_ensemble = np.clip(pred_ensemble, 0, 10)
        
        return round(pred_ensemble, 2)
    
    def save(self, base_path='ensemble_model'):
        """Guarda todos los modelos"""
        self.nn_model.save_model(f'{base_path}_nn.h5')
        
        with open(f'{base_path}_rf.pkl', 'wb') as f:
            pickle.dump(self.rf_model, f)
        
        with open(f'{base_path}_gb.pkl', 'wb') as f:
            pickle.dump(self.gb_model, f)
        
        print(f"\n✓ Modelos ensemble guardados:")
        print(f"  - {base_path}_nn.h5")
        print(f"  - {base_path}_rf.pkl")
        print(f"  - {base_path}_gb.pkl")
    
    def load(self, base_path='ensemble_model'):
        """Carga todos los modelos"""
        import os
        
        # Verificar que los archivos existen
        if not os.path.exists(f'{base_path}_nn.h5'):
            raise FileNotFoundError(f"No se encontró {base_path}_nn.h5. Primero entrena el modelo con ensemble_model.py")
        
        # Cargar Red Neuronal
        self.nn_model.load_model(f'{base_path}_nn.h5')
        
        # Cargar Random Forest
        with open(f'{base_path}_rf.pkl', 'rb') as f:
            self.rf_model = pickle.load(f)
        
        # Cargar Gradient Boosting
        with open(f'{base_path}_gb.pkl', 'rb') as f:
            self.gb_model = pickle.load(f)
        
        print(f"✓ Modelos ensemble cargados:")
        print(f"  - Red Neuronal")
        print(f"  - Random Forest")
        print(f"  - Gradient Boosting")


if __name__ == "__main__":
    # Entrenar ensemble
    ensemble = EnsembleAffinityModel()
    ensemble.train()
    ensemble.save()
    
    # Ejemplo de predicción
    print("\n" + "="*80)
    print("EJEMPLO DE PREDICCIÓN ENSEMBLE")
    print("="*80)
    
    job = """
    Desarrollador Senior Python - 5 años experiencia requerida
    Requisitos: Python, Django, PostgreSQL, Docker, Kubernetes
    Educación: Profesional en Ingeniería de Sistemas
    Inglés: Avanzado
    Modalidad: Remoto
    """
    
    resume = """
    Ingeniero de Software con 6 años de experiencia en desarrollo backend
    Skills: Python, Django, Flask, PostgreSQL, Docker, AWS, Git
    Educación: Profesional en Ingeniería de Sistemas
    Inglés: Avanzado
    """
    
    affinity = ensemble.predict(job, resume)
    print(f"\n🎯 Afinidad predicha (ENSEMBLE): {affinity}/10")
