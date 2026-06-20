import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Spam Guard Pro", page_icon="🛡️", layout="wide")

st.title("🛡️ Production-Grade AI Spam & Phishing Detector")
st.write("This application uses Natural Language Processing (NLP) to classify text messages based on patterns learned from thousands of real-world examples.")

# --- 1. LOAD DATASET (UPDATED NAME BREAKS CACHE) ---
@st.cache_data
def load_dataset_v2():
    data = pd.read_csv("spam_data.csv", encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
    return data

df = load_dataset_v2()

# --- 2. TRAIN PIPELINE (UPDATED NAME BREAKS CACHE) ---
@st.cache_resource
def train_model_v2(data):
    X = data['text']  
    y = data['label'] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Advanced Vectorization: Extracting both single words AND multi-word phrases (n-grams)
    ml_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))),
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    
    ml_pipeline.fit(X_train, y_train)
    
    predictions = ml_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return ml_pipeline, accuracy

model, model_accuracy = train_model_v2(df)

# --- 3. SIDEBAR METRICS ---
st.sidebar.header("📊 Model Metrics & Metadata")
st.sidebar.metric(label="Dataset Size", value=f"{len(df)} Rows")
st.sidebar.metric(label="Testing Accuracy", value=f"{model_accuracy * 100:.2f}%")
st.sidebar.markdown("""
**Technical Stack:**
* **Algorithm:** Multinomial Naive Bayes
* **Vectorization:** TF-IDF with Word Bi-grams
* **Tuning:** Parametric Smoothing (alpha=0.1)
""")

# --- 4. INTERACTIVE INTERFACE ---
st.subheader("Test the Production Model Live:")
user_input = st.text_area("Paste any suspicious email, SMS, or phishing message below:")

if st.button("Analyze with AI"):
    if user_input.strip() != "":
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]
        
        ham_confidence = probabilities[0] * 100
        spam_confidence = probabilities[1] * 100
        
        st.write("---")
        if prediction == 'spam':
            st.error(f"🚨 **System Alert: This looks like SPAM / PHISHING.**")
            st.write(f"AI Confidence Score: **{spam_confidence:.2f}%** probability of malicious intent.")
        else:
            st.success(f"✅ **System Verified: This looks like a legitimate message (HAM).**")
            st.write(f"AI Confidence Score: **{ham_confidence:.2f}%** probability of safety.")
            
        chart_data = pd.DataFrame({
            'Classification': ['Safe (Ham)', 'Spam'],
            'Probability (%)': [ham_confidence, spam_confidence]
        })
        st.bar_chart(data=chart_data, x='Classification', y='Probability (%)')
    else:
        st.warning("Please paste some text first!")