
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pytesseract
from pdf2image import convert_from_bytes
import cv_lstm_functions as cvf

print("Imports successful!")

try:
    nlp = cvf.load_spacy_model()
    print("Spacy model loaded.")
except Exception as e:
    print(f"Spacy load failed: {e}")

try:
    resume_model, tokenizer, encoder = cvf.load_model_artifacts(
        model_path="models/resume_classifier_model.h5",
        tokenizer_path="models/tokenizer .pickle",
        encoder_path="models/label_encoder.pkl",
    )
    print("Resume model loaded.")
except Exception as e:
    print(f"Resume model load failed: {e}")

try:
    job_model = cvf.load_job_model("models/job_recommender.h5")
    if job_model:
        print("Job model loaded.")
    else:
        print("Job model returned None.")
except Exception as e:
    print(f"Job model load failed: {e}")

print("Verification complete.")
