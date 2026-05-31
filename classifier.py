"""
classifier.py
-------------
Inference module for the SpamGuard spam detection system.

Loads the pre-trained DistilBERT-FGM classification model from the specified
checkpoint directory and exposes a single classify() function for synchronous
inference. The model and tokeniser are initialised once at module import time
to eliminate repeated I/O overhead on each inference request.

Model: DistilBERT-base-uncased fine-tuned with Fast Gradient Method (FGM)
       adversarial training on a multi-corpus dataset (Enron, LingSpam,
       SpamAssassin, TREC-07). Trained as part of a Master's thesis at
       Chaoyang University of Technology (CYUT), Taiwan.

Reference checkpoint: p3_full_fgm_ml256_s456
  - Training method : Fast Gradient Method (FGM) adversarial training
  - Max token length: 256
  - Training corpora: All four benchmark corpora (multi-corpus)
  - Random seed     : 456 (lowest cross-domain false positive rate)
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------
# The MODEL_PATH environment variable allows the path to be overridden at
# runtime (e.g. via docker-compose.yml). The default value targets the local
# development path on the project host machine.
MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    r'E:\V1_Upgraded_Thesis_Experiment\models\p3_full_fgm_ml256_s456\best'
)

# ---------------------------------------------------------------------------
# Model and tokeniser initialisation
# ---------------------------------------------------------------------------
# Both objects are instantiated at module level so that the ~255 MB weight
# file is loaded from disk exactly once per process lifetime. Loading inside
# the endpoint function would incur a multi-second delay on every request.
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Switch the model to evaluation mode to disable dropout layers, ensuring
# deterministic output during inference.
model.eval()

# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------
# These thresholds implement the Tier 2 decision boundary of the three-tier
# cascade architecture described in the thesis. Emails whose spam probability
# falls within the uncertainty band [THRESHOLD_HAM, THRESHOLD_SPAM] are
# classified as 'uncertain' and would normally escalate to Tier 3 (Flan-T5).
THRESHOLD_SPAM = 0.7
THRESHOLD_HAM  = 0.3


def classify(text: str) -> dict:
    """
    Perform binary spam classification on a single email text string.

    Tokenises the input, performs a forward pass through the DistilBERT-FGM
    model, and maps the resulting logits to a human-readable verdict.

    Parameters
    ----------
    text : str
        The raw email body text to be classified.

    Returns
    -------
    dict
        A dictionary containing:
        - result     (str)   : 'spam', 'ham', or 'uncertain'
        - confidence (float) : Probability of the spam class, rounded to 4
                               decimal places (range: 0.0 – 1.0)
        - tier_used  (int)   : Always 2 for this module, indicating that the
                               Tier 2 DistilBERT model produced the verdict
    """
    # Tokenise the input with truncation to the maximum sequence length used
    # during training (256 tokens). Longer inputs are silently truncated.
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=256
    )

    # Disable gradient computation during inference to reduce memory usage
    # and improve throughput on CPU-only deployments.
    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert raw logits to class probabilities via the softmax function.
    # Index 1 corresponds to the spam class as defined in config.json.
    probs  = torch.softmax(logits, dim=-1)[0]
    p_spam = float(probs[1])

    # Apply the cascade threshold logic to determine the final verdict.
    if p_spam > THRESHOLD_SPAM:
        verdict = 'spam'
    elif p_spam < THRESHOLD_HAM:
        verdict = 'ham'
    else:
        verdict = 'uncertain'

    return {
        'result'    : verdict,
        'confidence': round(p_spam, 4),
        'tier_used' : 2
    }
