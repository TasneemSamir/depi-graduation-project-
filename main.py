from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import fitz  # PyMuPDF

import cv_lstm_functions as cvf  # من مشروعك

# ===================== 1) إعداد FastAPI =====================

app = FastAPI(
    title="Resume LSTM API",
    description="API لتصنيف الـ CV باستخدام موديل الـ LSTM من المشروع",
    version="1.0.0"
)

# Add CORS Middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 2) تحميل الموديل والـ tokenizer والـ encoder و spaCy =====================

print("Loading model artifacts...")
resume_model, tokenizer, encoder = cvf.load_model_artifacts(
    model_path="models/resume_classifier_model.h5",
    tokenizer_path="models/tokenizer .pickle",   # بخلي الـ space لو انت مصرّ عليه
    encoder_path="models/label_encoder_corrected.pkl",
)

# Load Job Recommender Model
job_model = cvf.load_job_model("models/job_recommender.h5")

nlp = cvf.load_spacy_model()  # en_core_web_sm
print("Artifacts loaded successfully!")

# ===================== 3) Schemas =====================

class TextRequest(BaseModel):
    text: str

class TextPredictionResponse(BaseModel):
    predicted_category: str
    confidence: float

class PipelineResponse(BaseModel):
    extracted_text: str
    predicted_category: str
    confidence: float
    job_role_predictions: list
    summary: str

# ===================== 4) Prediction من نص عادي =====================

@app.post("/predict/resume_text", response_model=TextPredictionResponse)
def predict_resume_text(req: TextRequest):
    category, conf = cvf.predict_resume_category(
        resume_text=req.text,
        model=resume_model,
        tokenizer=tokenizer,
        encoder=encoder,
        max_length=500   # نفس الديفولت في الفنكشن بتاعتك
    )

    return TextPredictionResponse(
        predicted_category=category,
        confidence=conf
    )

# ===================== 5) Pipeline من PDF =====================

@app.post("/pipeline/analyze_resume", response_model=PipelineResponse)
async def analyze_resume(file: UploadFile = File(...)):
    # 1) قراءة الـ PDF using OCR
    content = await file.read()
    
    # Try OCR first
    full_text = cvf.extract_text_from_pdf_ocr(content)
    
    # Fallback to normal extraction if OCR returns empty (or you can combine both)
    if not full_text.strip():
        print("OCR returned empty, trying standard extraction...")
        with fitz.open(stream=content, filetype="pdf") as doc:
            pages_text = [page.get_text() for page in doc]
        full_text = "\n".join(pages_text).strip()

    if not full_text:
        full_text = "No text extracted from PDF."

    # 2) Prediction Category
    category, conf = cvf.predict_resume_category(
        resume_text=full_text,
        model=resume_model,
        tokenizer=tokenizer,
        encoder=encoder,
        max_length=500
    )
    
    # 3) Predict Job Role (Dynamic)
    job_predictions = cvf.predict_job_role(
        resume_text=full_text,
        model=job_model,
        tokenizer=tokenizer,
        encoder=encoder,
        max_length=500
    )

    summary = f"Predicted resume category: {category} (confidence = {conf:.2f})."

    return PipelineResponse(
        extracted_text=full_text,
        predicted_category=category,
        confidence=conf,
        job_role_predictions=job_predictions,
        summary=summary
    )
