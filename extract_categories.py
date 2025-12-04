
import pandas as pd

try:
    # Read only the Category column to save memory
    df = pd.read_csv('models/preprocessed_resumes.csv', usecols=['Category'])
    unique_categories = sorted(df['Category'].unique().tolist())
    
    print(f"Number of unique categories: {len(unique_categories)}")
    print("Categories:")
    for i, cat in enumerate(unique_categories):
        print(f"{i}: {cat}")
        
except Exception as e:
    print(f"Error reading CSV: {e}")
