# Milestone 1

Check README.md for an overview of the repo.

What are the questions or goals your project aims to answer?

What experiments should be done to answer that question, and how will you know from the outcome of the experiment that you have succeeded?

## Problem Overview

Discussion of existing implementations

Experiments to run:
- 

## Method

## Benchmarking

bar chart: Qwen Intern LlaVa legacy and fast vs runtime
- data: randomized 256,1048 images
- Hopefully also Multi-thread performance

bar chart: same but memory allocations

## Profiling

Profiling analysis


3 models

Evaluated at different ratios

Legacy vs Fast

Runtimes

Multiple Threads (?)



## Key Questions


- `phase0.py`: Self-contained sanity check file. Verifies that image pre-processing is slow.

- `phase1.py`: Pulls together `data.py`, `measurement.py`, and `models.py` to profile Qwen2.5-VL-7B

- `phase2.py`: Same as phase1, Profiles InternVL-2.5

## TODOS

- make sure we can properly profile the pytorch code
    - Implement profiling, run isolated test, get feedback
- Implement LlaVa AnyRes


- writeup