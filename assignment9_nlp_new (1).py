import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Hindi dataset
data = {
    "text": [
        "यह फिल्म बहुत अच्छी है",
        "मुझे यह खाना पसंद नहीं आया",
        "आज मौसम सामान्य है",
        "मोबाइल की बैटरी शानदार है",
        "सेवा बहुत खराब थी",
        "मुझे यह गाना अच्छा लगा",
        "यह उत्पाद बेकार है",
        "आज का दिन ठीक है",
        "शिक्षक बहुत अच्छे हैं",
        "मुझे यह ऐप पसंद नहीं है"
    ],

    "sentiment": [
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative"
    ]
}

# Convert into DataFrame
df = pd.DataFrame(data)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["text"])
y = df["sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom sentence
sample = ["यह मोबाइल बहुत अच्छा है"]

sample_vector = vectorizer.transform(sample)

prediction = model.predict(sample_vector)

print("Sentence:", sample[0])
print("Predicted Sentiment:", prediction[0])