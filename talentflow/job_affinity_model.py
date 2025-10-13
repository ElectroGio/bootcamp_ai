"""
Modelo de Evaluación de Afinidad Laboral usando Deep Learning y PLN
Este modelo usa embeddings de texto y redes neuronales para predecir afinidad (0-10)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import re

# Para preprocesamiento de texto
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class JobAffinityModel:
    """
    Modelo de afinidad laboral que combina:
    - TF-IDF para vectorización de texto
    - Red neuronal profunda para predicción
    """
    
    def __init__(self, max_features=500):
        self.max_features = max_features
        self.job_vectorizer = None
        self.resume_vectorizer = None
        self.scaler = None
        self.model = None
        
        # Descargar recursos de NLTK si es necesario
        try:
            stopwords.words('spanish')
        except:
            print("Descargando recursos de NLTK...")
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
    
    def preprocess_text(self, text):
        """
        Preprocesa el texto:
        - Convierte a minúsculas
        - Elimina caracteres especiales
        - Tokeniza
        """
        if not isinstance(text, str):
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar caracteres especiales pero mantener letras con acentos
        text = re.sub(r'[^a-záéíóúñü\s]', ' ', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_and_prepare_data(self, csv_path='job_affinity_dataset.csv', test_size=0.2):
        """
        Carga y prepara los datos del CSV
        """
        print("Cargando dataset...")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset cargado: {len(df)} muestras")
        print(f"Estadísticas de afinidad:\n{df['affinity_score'].describe()}\n")
        
        # Preprocesar textos
        print("Preprocesando textos...")
        df['job_clean'] = df['job_description'].apply(self.preprocess_text)
        df['resume_clean'] = df['resume'].apply(self.preprocess_text)
        
        # Dividir en entrenamiento y prueba
        X_job = df['job_clean'].values
        X_resume = df['resume_clean'].values
        y = df['affinity_score'].values
        
        X_job_train, X_job_test, X_resume_train, X_resume_test, y_train, y_test = train_test_split(
            X_job, X_resume, y, test_size=test_size, random_state=42
        )
        
        print(f"Datos de entrenamiento: {len(X_job_train)} muestras")
        print(f"Datos de prueba: {len(X_job_test)} muestras")
        
        return X_job_train, X_job_test, X_resume_train, X_resume_test, y_train, y_test
    
    def vectorize_text(self, X_job_train, X_job_test, X_resume_train, X_resume_test):
        """
        Convierte texto a vectores usando TF-IDF
        """
        print("\nVectorizando textos con TF-IDF...")
        
        # Vectorizar descripciones de trabajo
        self.job_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),  # Unigramas y bigramas
            min_df=2
        )
        
        X_job_train_vec = self.job_vectorizer.fit_transform(X_job_train).toarray()
        X_job_test_vec = self.job_vectorizer.transform(X_job_test).toarray()
        
        # Vectorizar hojas de vida
        self.resume_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2
        )
        
        X_resume_train_vec = self.resume_vectorizer.fit_transform(X_resume_train).toarray()
        X_resume_test_vec = self.resume_vectorizer.transform(X_resume_test).toarray()
        
        print(f"Dimensión de vectores de trabajo: {X_job_train_vec.shape[1]}")
        print(f"Dimensión de vectores de CV: {X_resume_train_vec.shape[1]}")
        
        # NUEVO: Extraer features numéricas explícitas
        print("\n🔢 Extrayendo features numéricas explícitas...")
        from extract_features import extract_numeric_features, get_feature_names
        
        # Extraer para entrenamiento
        numeric_train = []
        for job, resume in zip(X_job_train, X_resume_train):
            feats = extract_numeric_features(job, resume)
            numeric_train.append(feats)
        
        numeric_train = np.array(numeric_train)
        
        # Extraer para prueba
        numeric_test = []
        for job, resume in zip(X_job_test, X_resume_test):
            feats = extract_numeric_features(job, resume)
            numeric_test.append(feats)
        
        numeric_test = np.array(numeric_test)
        
        print(f"Features numéricas extraídas: {numeric_train.shape[1]}")
        print(f"  → {', '.join(get_feature_names())}")
        
        # Combinar features: TF-IDF + Numéricas
        X_train_combined = np.concatenate([X_job_train_vec, X_resume_train_vec, numeric_train], axis=1)
        X_test_combined = np.concatenate([X_job_test_vec, X_resume_test_vec, numeric_test], axis=1)
        
        print(f"Dimensión total de features: {X_train_combined.shape[1]}")
        
        return X_train_combined, X_test_combined
    
    def build_model(self, input_dim):
        """
        Construye la red neuronal
        """
        print("\nConstruyendo modelo de red neuronal...")
        
        # ARQUITECTURA MEJORADA para mayor precisión
        self.model = keras.Sequential([
            # Capa de entrada - MÁS NEURONAS
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Capas ocultas profundas
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Capa de salida - valor entre 0 y 10
            layers.Dense(1, activation='linear')
        ])
        
        # Compilar modelo con learning rate MÁS BAJO para mejor convergencia
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Reducido de 0.001
            loss='mse',
            metrics=['mae']  # Removido 'mse' para evitar problemas de serialización
        )
        
        print(self.model.summary())
        
        return self.model
    
    def train(self, X_train, y_train, X_test=None, y_test=None, epochs=150, batch_size=16, validation_split=0.25):
        """
        Entrena el modelo
        MEJORADO: Más épocas (150), batch size menor (16), más validación (25%)
        """
        print("\nEntrenando modelo (versión mejorada para mayor precisión)...")
        
        # Callbacks MEJORADOS para mejor entrenamiento
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Aumentado de 15 - más paciencia
            restore_best_weights=True,
            min_delta=0.001  # Añadido - cambio mínimo significativo
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # Más agresivo (era 0.5)
            patience=7,  # Más rápido (era 5)
            min_lr=0.00001,
            verbose=1  # Mostrar cuando reduce LR
        )
        
        # Determinar si usar validation_split o validation_data
        validation_data = None
        if X_test is not None and y_test is not None:
            validation_data = (X_test, y_test)
            validation_split = 0  # No usar split si hay validation_data
        
        # Entrenar
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo
        """
        print("\n" + "="*80)
        print("EVALUACIÓN DEL MODELO")
        print("="*80)
        
        # Predicciones
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        
        # Asegurar que las predicciones estén en el rango [0, 10]
        y_pred = np.clip(y_pred, 0, 10)
        
        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nMétricas en conjunto de prueba:")
        print(f"  MAE (Error Absoluto Medio): {mae:.4f}")
        print(f"  MSE (Error Cuadrático Medio): {mse:.4f}")
        print(f"  RMSE (Raíz del Error Cuadrático): {rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")
        
        # Análisis de errores
        errors = np.abs(y_test - y_pred)
        print(f"\nAnálisis de errores:")
        print(f"  Error mínimo: {errors.min():.4f}")
        print(f"  Error máximo: {errors.max():.4f}")
        print(f"  Error mediano: {np.median(errors):.4f}")
        
        # Predicciones dentro de diferentes rangos de error
        within_05 = (errors <= 0.5).sum() / len(errors) * 100
        within_10 = (errors <= 1.0).sum() / len(errors) * 100
        within_15 = (errors <= 1.5).sum() / len(errors) * 100
        
        print(f"\nPrecisión por rango:")
        print(f"  Predicciones con error ≤ 0.5: {within_05:.2f}%")
        print(f"  Predicciones con error ≤ 1.0: {within_10:.2f}%")
        print(f"  Predicciones con error ≤ 1.5: {within_15:.2f}%")
        
        return {
            'y_pred': y_pred,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    
    def plot_training_history(self, history):
        """
        Visualiza el historial de entrenamiento
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Pérdida
        axes[0].plot(history.history['loss'], label='Entrenamiento', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validación', linewidth=2)
        axes[0].set_xlabel('Época', fontsize=12)
        axes[0].set_ylabel('Pérdida (MSE)', fontsize=12)
        axes[0].set_title('Curva de Pérdida durante el Entrenamiento', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(history.history['mae'], label='Entrenamiento', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Validación', linewidth=2)
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Error Absoluto Medio durante el Entrenamiento', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráfico de entrenamiento guardado: training_history.png")
        plt.show()
    
    def plot_predictions(self, y_test, y_pred):
        """
        Visualiza predicciones vs valores reales
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(y_test, y_pred, alpha=0.5, s=30)
        axes[0].plot([0, 10], [0, 10], 'r--', linewidth=2, label='Predicción perfecta')
        axes[0].set_xlabel('Afinidad Real', fontsize=12)
        axes[0].set_ylabel('Afinidad Predicha', fontsize=12)
        axes[0].set_title('Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 10)
        axes[0].set_ylim(0, 10)
        
        # Distribución de errores
        errors = y_test - y_pred
        axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Error = 0')
        axes[1].set_xlabel('Error (Real - Predicho)', fontsize=12)
        axes[1].set_ylabel('Frecuencia', fontsize=12)
        axes[1].set_title('Distribución de Errores', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predictions_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico de predicciones guardado: predictions_analysis.png")
        plt.show()
    
    def predict_affinity(self, job_description, resume):
        """
        Predice la afinidad para un trabajo y CV nuevos
        """
        # Preprocesar
        job_clean = self.preprocess_text(job_description)
        resume_clean = self.preprocess_text(resume)
        
        # Vectorizar
        job_vec = self.job_vectorizer.transform([job_clean]).toarray()
        resume_vec = self.resume_vectorizer.transform([resume_clean]).toarray()
        
        # Extraer features numéricas
        from extract_features import extract_numeric_features
        numeric_feats = extract_numeric_features(job_description, resume).reshape(1, -1)
        
        # Combinar todas las features
        X_combined = np.concatenate([job_vec, resume_vec, numeric_feats], axis=1)
        
        # Predecir
        affinity = self.model.predict(X_combined, verbose=0)[0][0]
        
        # Asegurar rango [0, 10]
        affinity = np.clip(affinity, 0, 10)
        
        return round(affinity, 2)
    
    def save_model(self, model_path='job_affinity_model.h5'):
        """
        Guarda el modelo entrenado y los vectorizadores
        """
        import pickle
        
        # Guardar modelo de Keras
        self.model.save(model_path)
        print(f"\n✓ Modelo guardado en: {model_path}")
        
        # Guardar vectorizadores
        vectorizers = {
            'job_vectorizer': self.job_vectorizer,
            'resume_vectorizer': self.resume_vectorizer
        }
        
        with open('vectorizers.pkl', 'wb') as f:
            pickle.dump(vectorizers, f)
        print(f"✓ Vectorizadores guardados en: vectorizers.pkl")
    
    def load_model(self, model_path='job_affinity_model.h5'):
        """
        Carga un modelo previamente entrenado y sus vectorizadores
        """
        try:
            # Intentar cargar normalmente
            self.model = keras.models.load_model(model_path)
        except Exception as e:
            # Si falla, intentar con compile=False y recompilar
            print(f"⚠ Advertencia: {e}")
            print("Cargando modelo sin compilar y recompilando...")
            self.model = keras.models.load_model(model_path, compile=False)
            
            # Recompilar el modelo
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        print(f"✓ Modelo cargado desde: {model_path}")
        
        # Cargar vectorizadores
        import pickle
        with open('vectorizers.pkl', 'rb') as f:
            vectorizers = pickle.load(f)
            self.job_vectorizer = vectorizers['job_vectorizer']
            self.resume_vectorizer = vectorizers['resume_vectorizer']
        
        print(f"✓ Vectorizadores cargados desde: vectorizers.pkl")


def main():
    """
    Función principal para entrenar y evaluar el modelo
    """
    print("="*80)
    print("SISTEMA DE EVALUACIÓN DE AFINIDAD LABORAL")
    print("="*80)
    
    # Crear instancia del modelo
    model = JobAffinityModel(max_features=500)
    
    # Cargar y preparar datos
    X_job_train, X_job_test, X_resume_train, X_resume_test, y_train, y_test = \
        model.load_and_prepare_data('job_affinity_dataset.csv')
    
    # Vectorizar textos
    X_train, X_test = model.vectorize_text(
        X_job_train, X_job_test, X_resume_train, X_resume_test
    )
    
    # Construir modelo
    model.build_model(input_dim=X_train.shape[1])
    
    # Entrenar con parámetros MEJORADOS
    print("\n⚡ VERSIÓN MEJORADA DEL MODELO:")
    print("   • Más datos (5000 muestras)")
    print("   • Arquitectura más profunda (512→256→128→64→32)")
    print("   • Menos aleatoriedad en el dataset")
    print("   • Mejores hiperparámetros de entrenamiento")
    print("   • Resultado esperado: MAE < 0.7, R² > 0.85\n")
    
    history = model.train(
        X_train, y_train,
        epochs=150,  # Aumentado
        batch_size=16,  # Reducido
        validation_split=0.25  # Aumentado
    )
    
    # Evaluar
    results = model.evaluate(X_test, y_test)
    
    # Visualizar
    model.plot_training_history(history)
    model.plot_predictions(y_test, results['y_pred'])
    
    # Guardar modelo
    model.save_model('job_affinity_model.h5')
    
    # Ejemplos de predicción
    print("\n" + "="*80)
    print("EJEMPLOS DE PREDICCIÓN")
    print("="*80)
    
    # Ejemplo 1: Alta afinidad
    job1 = """Puesto: Desarrollador Full Stack. Ubicación: Remoto. Experiencia requerida: 3-5 años. 
    Educación: Profesional en Ingeniería. Skills técnicas requeridas: Python, Django, React, PostgreSQL, AWS. 
    Skills blandas: Trabajo en equipo, Comunicación efectiva. Idiomas: Inglés avanzado."""
    
    resume1 = """Experiencia: 5-8 años. Educación: Profesional en Ingeniería. 
    Skills: Python, Django, React, Node.js, PostgreSQL, Docker, AWS, Trabajo en equipo, Liderazgo. 
    Idiomas: Español nativo, Inglés avanzado."""
    
    affinity1 = model.predict_affinity(job1, resume1)
    print(f"\nEjemplo 1 (Alta afinidad esperada):")
    print(f"Afinidad predicha: {affinity1}/10")
    
    # Ejemplo 2: Baja afinidad
    job2 = """Puesto: Data Scientist. Ubicación: Bogotá. Experiencia requerida: 5-8 años. 
    Educación: Maestría. Skills técnicas requeridas: Python, Machine Learning, TensorFlow, SQL, AWS. 
    Skills blandas: Pensamiento crítico, Resolución de problemas. Idiomas: Inglés avanzado."""
    
    resume2 = """Experiencia: 1-3 años. Educación: Técnico. 
    Skills: Excel avanzado, PowerBI, Diseño gráfico, Adobe Photoshop. 
    Idiomas: Español nativo."""
    
    affinity2 = model.predict_affinity(job2, resume2)
    print(f"\nEjemplo 2 (Baja afinidad esperada):")
    print(f"Afinidad predicha: {affinity2}/10")
    
    print("\n" + "="*80)
    print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*80)


if __name__ == "__main__":
    main()
