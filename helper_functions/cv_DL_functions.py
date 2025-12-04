import re
import pickle
import warnings

from tqdm.auto import tqdm

from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')



def clean_resume_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text.lower()

def process_with_spacy(df, nlp):
    print("Processing text with spaCy (lemmatization & POS tagging)...")

    texts = df['Resume'].tolist()
    processed_texts = []

    docs = nlp.pipe(texts, disable=["parser", "ner"], batch_size=50, n_process=-1)

    for doc in tqdm(docs, total=len(texts), desc="Processing Resumes"):
        processed_texts.append(' '.join([
            f"{token.lemma_}_{token.pos_}" for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]))

    df['Resume_POS_text'] = processed_texts

    print("spaCy processing with POS tags completed.")
    return df

def load_models(model_path='resume_classifier_DL_model.keras',
                vectorizer_path='tfidf_vectorizer.pkl',
                label_encoder_path='label_encoder.pkl'):
    
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")


    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"Vectorizer loaded from {vectorizer_path}!")

    with open(label_encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    print(f"Label Encoder loaded from {label_encoder_path}!")

    return model, vectorizer, encoder