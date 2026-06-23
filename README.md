# 🛡️ Production-Grade AI Spam & Phishing Detector

A high-performance, responsive web application that detects malicious text messages, phishing emails, and fraudulent spam inputs. This project features a **hybrid production architecture** that combines machine learning classification with a custom heuristic validation layer to handle complex adversarial inputs.


---

## 🚀 Key Features

* **Hybrid System Architecture:** Combines a statistical machine learning model with an enterprise-grade regex heuristic layer to protect against sneaky adversarial attacks.
* **Serverless Deployment (WebAssembly):** Powered by `stlite` to run Python, Streamlit, and Scikit-Learn natively inside the client's browser, eliminating hosting costs and server latency.
* **Phrase-Level Intelligence:** Integrated word bi-grams ($ngram\_range=(1, 2)$) to identify contextual phrase combinations (e.g., "bank details", "won prize") instead of just isolated tokens.
* **Interactive Data Visualization:** Instantly charts the predictive confidence split between safe (Ham) and unsafe (Spam) classifications.

---

## 🧠 Architectural Overview & Technical Stack

In real-world cybersecurity ecosystems, relying solely on standalone machine learning models leaves apps vulnerable to adversarial text (e.g., when spam phrases are buried inside high-frequency "friendly" words like *"please share"*). 

This project implements a multi-tiered security pipeline:
1. **The Heuristic Pre-Filter:** Intercepts incoming strings to check for high-risk financial fraud strings and unverified high-value reward claims (e.g., regional phrases like "1cr", "lakh", or "$").
2. **The ML Core Pipeline (Fallback):** If no hard rules are triggered, a Multinomial Naive Bayes model processes the text through a TF-IDF Vectorizer tuned with Laplace smoothing ($\alpha=0.1$) to deliver high-precision probability scores.

### Technologies Used:
* **Frontend/UI:** Streamlit / `stlite` (WebAssembly implementation)
* **Machine Learning Framework:** Scikit-Learn (Pipelines, Feature Extraction, Model Selection)
* **Data Processing:** Pandas
* **Core Language:** Python 3

---

## 📊 Model Performance

The underlying classifier trains on a standard dataset of 5,572 real-world text interactions:
* **Training Accuracy:** ~98.65%
* **Feature Extraction:** Word-level Tokenization with Stop-Word Filtering and sub-linear TF scaling.

---


