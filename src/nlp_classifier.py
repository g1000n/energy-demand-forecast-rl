from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "Energy demand increased",
    "Power usage decreased",
    "Electricity stable"
]

labels = ["increase", "decrease", "neutral"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

print("NLP prototype ready")