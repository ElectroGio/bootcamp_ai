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
#  python -m uvicorn api_talentflow:app --reload --port 8000
# Luego ir a: http://127.0.0.1:8000/docs para probar por Swagger.
