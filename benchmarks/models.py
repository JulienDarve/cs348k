from transformers import AutoImageProcessor


def load_processors(model_id):
    """Return (slow_processor, fast_processor) for the given HuggingFace model_id."""
    slow = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    fast = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    return slow, fast
