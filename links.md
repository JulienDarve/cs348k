# Links for VLM Pre-Processing Pipelines

Links and citations for model papers, code, and documentation.

## Halide

https://people.csail.mit.edu/jrk/halide12/halide12.pdf

The original Halide paper introduces a language for image-processing pipelines that separates the algorithm from the schedule used to optimize parallelism, locality, vectorization, and recomputation. It is directly relevant to this project because the Milestone 2 kernel work is inspired by Halide-style scheduling ideas such as fusion, tiling, and moving computation closer to where values are consumed.

## Llava-NeXT AnyRes

### Original blog + paper

https://arxiv.org/pdf/2408.03326

LLaVA-NeXT technical report describing the AnyRes architecture and benchmark results.

https://llava-vl.github.io/blog/2024-01-30-llava-next/

Project blog post introducing LLaVA-NeXT and its higher-resolution image handling.

### Huggingface code on github

https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llava_next/image_processing_llava_next.py

Reference Hugging Face Python image processor for the LLaVA-NeXT preprocessing pipeline.

https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llava_next/image_processing_llava_next_fast.py

Fast Hugging Face image processor implementation for LLaVA-NeXT preprocessing.

### Huggingface Model Card

https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf

Model card for the Hugging Face LLaVA-NeXT Mistral 7B checkpoint used as a representative AnyRes model.

## InternVL 3.5

### InternVL 2.5 blog and paper

https://internvl.github.io/blog/2024-12-05-InternVL-2.5/

Project blog post introducing InternVL 2.5 and its dynamic high-resolution vision pipeline.

https://arxiv.org/pdf/2412.05271

InternVL 2.5 paper with model architecture, training recipe, and evaluation details.

### InternVL 3.5 blog and paper

https://internvl.github.io/blog/2025-08-26-InternVL-3.5/

Project blog post announcing InternVL 3.5 and summarizing improvements over the 2.5 series.

https://arxiv.org/abs/2508.18265

InternVL 3.5 paper page covering the updated model family and evaluation results.

### InternVL 3.5 huggingface github code

https://github.com/huggingface/transformers/blob/main/src/transformers/models/internvl/processing_internvl.py

Hugging Face processor wrapper for InternVL inputs and tokenizer/image-processor coordination.

https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/got_ocr2/image_processing_got_ocr2.py

Reference Hugging Face image processor code for the GOT-OCR2 vision preprocessing path reused by InternVL-related processors.

https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/got_ocr2/image_processing_got_ocr2_fast.py

Fast Hugging Face image processor code for the GOT-OCR2 preprocessing path.

### Huggingface model card

https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF

Model card for the Hugging Face InternVL3.5 8B checkpoint.

### InternVL github page

https://github.com/OpenGVLab/InternVL

Main InternVL repository with model code, examples, and release notes from OpenGVLab.

## Qwen2.5-VL

### Original blog + paper

https://qwen.ai/blog?id=qwen2.5-vl 

Official Qwen blog post introducing Qwen2.5-VL and its visual localization, document understanding, and video capabilities.

https://arxiv.org/abs/2502.13923

Qwen2.5-VL technical report describing the architecture, dynamic resolution processing, and evaluations.

### Huggingface code on github

https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_5_vl/image_processing_qwen2_5_vl.py

Reference Hugging Face image processor for Qwen2.5-VL image and video preprocessing.

https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_5_vl/image_processing_qwen2_5_vl_fast.py

Fast Hugging Face image processor implementation for Qwen2.5-VL preprocessing.

https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py

Hugging Face processor wrapper that combines Qwen2.5-VL image/video processing with tokenization.

### Huggingface model card

https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

Model card for the Qwen2.5-VL 7B instruct checkpoint commonly used for experiments.

### Qwen2.5-VL github page

https://github.com/QwenLM/Qwen2.5-VL

Official Qwen2.5-VL repository with inference examples, utilities, and links to released checkpoints.
