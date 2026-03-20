from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "Energy demand increased",
    "Power usage decreased",
    "Electricity demand is high",
    "Energy consumption dropped",
    "Electricity usage is stable"
]

labels = ["increase", "decrease", "increase", "decrease", "neutral"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

# test prediction
sample = ["power demand increased significantly"]
prediction = model.predict(vectorizer.transform(sample))

print("Prediction:", prediction[0])
print("NLP prototype ready")