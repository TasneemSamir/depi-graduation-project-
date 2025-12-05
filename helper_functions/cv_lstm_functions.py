import re
import pickle
import warnings
from tqdm.auto import tqdm

from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')


def load_spacy_model(model_name='en_core_web_sm'):

    try:
        nlp = spacy.load(model_name)
        print(f"spaCy model '{model_name}' loaded successfully")
        return nlp
    except OSError:
        print(f"Downloading spaCy model '{model_name}'...")
        import os
        os.system(f"python -m spacy download {model_name}")
        nlp = spacy.load(model_name)
        return nlp
        
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

def load_models(model_path='resume_classifier_model.keras',
                        tokenizer_path='tokenizer.pickle',
                        encoder_path='label_encoder.pickle'):

    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print(f"Tokenizer loaded from {tokenizer_path}")

    with open(encoder_path, 'rb') as handle:
        encoder = pickle.load(handle)
    print(f"Label encoder loaded from {encoder_path}")

    return model, tokenizer, encoder
