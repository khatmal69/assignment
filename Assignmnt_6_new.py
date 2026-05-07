# =========================================================
# MACHINE TRANSLATION SYSTEM
# English → Hindi Translation using MarianMT
# =========================================================

# =========================================================
# STEP 1 — INSTALL REQUIRED LIBRARIES
# =========================================================
# Run this in Jupyter/Colab
#if running in .py run the below line in terminal
!pip install transformers sentencepiece sacrebleu gradio torch

# =========================================================
# STEP 2 — IMPORT LIBRARIES
# =========================================================
from transformers import MarianMTModel, MarianTokenizer
from sacrebleu import corpus_bleu
import torch

# =========================================================
# STEP 3 — LOAD PRE-TRAINED MODEL
# =========================================================
# Helsinki-NLP English to Hindi Model
model_name = "Helsinki-NLP/opus-mt-en-hi"

# Load tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# OPTIMIZATION: Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Model Loaded Successfully on {device.upper()}!")

# =========================================================
# STEP 4 — CREATE TRANSLATION FUNCTION
# =========================================================
def translate_text(text):
    # Tokenize input text and move to device (CPU/GPU)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Generate translation (added max_length to prevent warnings)
    translated = model.generate(**inputs, max_length=512)

    # Decode translated tokens
    translated_text = tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    return translated_text

# =========================================================
# STEP 5 — TEST ON SINGLE SENTENCE
# =========================================================
text = "Welcome to the Smart City Traffic Management System"
print("Original:", text)
print("Translated:", translate_text(text))
print("=" * 50)

# =========================================================
# STEP 6 — TEST ON MULTIPLE SENTENCES
# =========================================================
sentences = [
    "Emergency services are available 24 hours",
    "Follow traffic rules for safety",
    "Public transport is eco-friendly"
]

for s in sentences:
    print("EN:", s)
    print("HI:", translate_text(s))
    print("-" * 40)

# =========================================================
# STEP 7 — EVALUATE USING BLEU SCORE
# =========================================================
# Reference translation (The ground truth)
reference = ["स्मार्ट सिटी में आपका स्वागत है"]

# Candidate translation from model (What your model predicted)
candidate = [translate_text("Welcome to Smart City")]

# Calculate BLEU score (sacrebleu expects a list of candidates and a list of reference lists)
score = corpus_bleu(candidate, [reference])

print("BLEU Score:", score.score)