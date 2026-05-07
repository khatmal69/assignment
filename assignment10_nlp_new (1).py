import nltk
from nltk.corpus import brown
from nltk.util import ngrams
from collections import defaultdict, Counter

# Download corpus
nltk.download('brown')

# -----------------------------------
# LOAD AND PREPROCESS CORPUS
# -----------------------------------

words = brown.words()

# Keep only alphabetic words and convert to lowercase
tokens = [word.lower() for word in words if word.isalpha()]

# -----------------------------------
# BUILD TRIGRAM MODEL
# -----------------------------------

trigrams = ngrams(tokens, 3)

model = defaultdict(list)

for w1, w2, w3 in trigrams:
    model[(w1, w2)].append(w3)

# -----------------------------------
# PREDICTION FUNCTION
# -----------------------------------

def predict_next(word1, word2, top_n=5):

    key = (word1.lower(), word2.lower())

    print(f"\nInput Words: {word1} {word2}")

    if key in model:

        predictions = Counter(model[key])

        print("\nTop Predictions:\n")

        total = sum(predictions.values())

        for word, freq in predictions.most_common(top_n):

            probability = freq / total

            print(f"{word} ---> Probability: {probability:.3f}")

    else:
        print("No predictions found")


# -----------------------------------
# EXAMPLE PREDICTIONS
# -----------------------------------

predict_next("in", "the")

predict_next("i", "am")

predict_next("it", "was")