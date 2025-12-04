
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

try:
    print("Reading categories from CSV...")
    # Read only the Category column
    df = pd.read_csv('models/preprocessed_resumes.csv', usecols=['Category'])
    unique_categories = sorted(df['Category'].unique().tolist())
    
    print(f"Found {len(unique_categories)} unique categories.")
    
    # Create and fit encoder
    encoder = LabelEncoder()
    encoder.fit(unique_categories)
    
    # Save encoder
    output_path = 'models/label_encoder_corrected.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(encoder, f)
        
    print(f"Corrected encoder saved to {output_path}")
    
    # Verify
    print(f"Encoder classes: {len(encoder.classes_)}")
    print(f"Class 73: {encoder.inverse_transform([73])[0]}")
    print(f"Class 87: {encoder.inverse_transform([87])[0]}")

except Exception as e:
    print(f"Error: {e}")
