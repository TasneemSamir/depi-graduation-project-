
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    model = load_model('models/job_recommender.h5')
    print("Model: job_recommender.h5")
    model.summary()
except Exception as e:
    print(f"Error loading job_recommender.h5: {e}")

try:
    model2 = load_model('models/model.h5')
    print("\nModel: model.h5")
    model2.summary()
except Exception as e:
    print(f"Error loading model.h5: {e}")
