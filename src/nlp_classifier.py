from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Simplified training for the prototype
texts = ["demand increased", "usage decreased", "stable power"]
labels = ["increase", "decrease", "neutral"]

vectorizer = TfidfVectorizer()
X_nlp = vectorizer.fit_transform(texts)
clf = LogisticRegression().fit(X_nlp, labels)

def get_nlp_multiplier(alert_text):
    pred = clf.predict(vectorizer.transform([alert_text]))[0]
    return {"increase": 1.5, "decrease": 0.7, "neutral": 1.0}.get(pred, 1.0)