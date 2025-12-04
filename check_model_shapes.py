
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    resume_model = load_model('models/resume_classifier_model.h5')
    print(f"Resume Model Output Shape: {resume_model.output_shape}")
except Exception as e:
    print(f"Error loading resume model: {e}")

try:
    job_model = load_model('models/job_recommender.h5')
    print(f"Job Model Output Shape: {job_model.output_shape}")
except Exception as e:
    print(f"Error loading job model: {e}")
