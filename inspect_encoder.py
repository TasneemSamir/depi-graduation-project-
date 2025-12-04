
import pickle
import numpy as np

try:
    with open('models/label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    print(f"Encoder type: {type(encoder)}")
    if hasattr(encoder, 'classes_'):
        print(f"Number of classes: {len(encoder.classes_)}")
        print(f"First 10 classes: {encoder.classes_[:10]}")
        print(f"Last 10 classes: {encoder.classes_[-10:]}")
    else:
        print("Encoder has no 'classes_' attribute.")

except Exception as e:
    print(f"Error loading encoder: {e}")
