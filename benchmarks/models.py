import math

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoImageProcessor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_ID_QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_ID_INTERNVL = "OpenGVLab/InternVL2_5-8B"
MODEL_ID_INTERNVL35 = "OpenGVLab/InternVL3-8B-hf"
MODEL_ID_LLAVA = "llava-hf/llava-v1.6-mistral-7b-hf"


def load_processors(model_id, **kwargs):
    """Return (slow_processor, fast_processor) for the given HuggingFace model_id."""
    slow = AutoImageProcessor.from_pretrained(model_id, use_fast=False, **kwargs)
    fast = AutoImageProcessor.from_pretrained(model_id, use_fast=True, **kwargs)
    return slow, fast


def get_internvl35_hf_processor():
    """Return the HF image processor for InternVL3.5-8B (GotOcr2ImageProcessorFast).

    crop_to_patches=True is the default set by InternVLProcessor's kwargs but
    is False in the standalone preprocessor_config.json, so it is passed explicitly.
    """
    proc = AutoImageProcessor.from_pretrained(MODEL_ID_INTERNVL35)

    def process_batch(images):
        return proc(images, crop_to_patches=True, return_tensors="pt")

    return process_batch


# --- verbatim from OpenGVLab/InternVL2_5-8B model card Quickstart ---

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# --- end verbatim ---


def get_internvl_manual_processor(max_num=12):
    """Return a process_batch(images) callable using the verbatim model-card pipeline."""
    transform = build_transform(input_size=448)

    def process_batch(images):
        pixel_values_list = []
        for image in images:
            tiles = dynamic_preprocess(image, min_num=1, max_num=max_num,
                                       image_size=448, use_thumbnail=True)
            pixel_values_list.append(torch.stack([transform(tile) for tile in tiles]))
        return {"pixel_values": torch.cat(pixel_values_list, dim=0)}

    return process_batch


def get_llava_hf_processor(use_fast=False):
    """Return the HF AutoImageProcessor for llava-v1.6-mistral-7b-hf."""
    return AutoImageProcessor.from_pretrained(MODEL_ID_LLAVA, use_fast=use_fast)
