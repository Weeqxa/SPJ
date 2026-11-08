
import re

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
from transformers import AutoModel, AutoTokenizer  # pyright: ignore[reportMissingImports]


MODEL_NAME = "bert-base-uncased"


def load_bert_model():
    """
    Load a pretrained BERT model and its tokenizer.

    Returns:
        tokenizer: BERT tokenizer.
        model: Pretrained BERT model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    model = AutoModel.from_pretrained(MODEL_NAME)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print(f"BERT model loaded: {MODEL_NAME}")
    print(f"Using device: {device}")

    return tokenizer, model


def get_contextual_embedding(
    sentence: str,
    target_word: str,
    tokenizer,
    model
) -> np.ndarray:
    """
    Extract a contextual embedding of a target word
    from a sentence using BERT.

    Args:
        sentence: Input sentence.
        target_word: Word whose embedding we want to extract.
        tokenizer: Loaded BERT tokenizer.
        model: Loaded BERT model.

    Returns:
        Contextual embedding as a NumPy array.
    """

    # Find the target word in the original sentence.
    pattern = rf"(?<!\w){re.escape(target_word)}(?!\w)"

    matches = list(
        re.finditer(pattern, sentence, re.IGNORECASE)
    )

    if not matches:
        raise ValueError(
            f"Word '{target_word}' was not found in the sentence."
        )

    # Use the first occurrence of the target word.
    start, end = matches[0].span()

    # Tokenize the sentence and preserve character offsets.
    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True
    )

    offset_mapping = encoded.pop("offset_mapping")[0].tolist()

    # Find tokens corresponding to the target word.
    target_token_indices = [
        index
        for index, (token_start, token_end) in enumerate(offset_mapping)
        if token_start < end and token_end > start
    ]

    if not target_token_indices:
        raise ValueError(
            f"Could not find tokens for word '{target_word}'. "
            "The word may be outside the model's input limit."
        )

    # Move input tensors to the model's device.
    device = next(model.parameters()).device

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    # Generate contextual representations.
    with torch.inference_mode():
        outputs = model(**encoded)

    # Extract hidden states of the target word.
    hidden_states = outputs.last_hidden_state

    target_embeddings = hidden_states[
        0, target_token_indices, :
    ]

    # Average subtoken representations if necessary.
    word_embedding = target_embeddings.mean(dim=0)

    return word_embedding.cpu().numpy()
