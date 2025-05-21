# Numerical Token Integrity Loss (NTIL)

NTIL is a specialized loss function designed to improve numerical prediction accuracy in language models. It addresses the limitations of traditional cross-entropy loss when dealing with numerical values by considering the semantic relationships between numbers.

## Overview

Traditional language models trained with cross-entropy loss often struggle with numerical reasoning and precision. NTIL solves this problem by:

1. Replacing token-level cross-entropy with Earth Mover's Distance (EMD) to better capture numerical relationships
2. Applying exponential position-based weighting to respect place-value number systems
3. Evaluating numerical differences holistically at the sequence level
4. Combining multiple loss components to capture various aspects of numerical similarity

## Features

- **Token-level EMD Loss**: Measures the distance between predicted and actual digit distributions
- **Exponential Position-based Weighting**: Applies higher weights to more significant digit positions
- **Absolute Numerical Loss**: Computes relative difference between predicted and actual numerical values
- **Order of Magnitude Loss**: Penalizes predictions that differ in scale from the target values
- **Seamless Integration**: Works alongside traditional MLE/cross-entropy loss

## Parameters

Parameters for NTIL:
- **tokenizer**: HuggingFace tokenizer for identifying digit tokens
- **batch_size**: Batch size for processing (to avoid number leakage between batches)
- **ntil_lambda**: Overall weight of NTIL relative to MLE loss
- **abs_alpha**: Weight for absolute numerical difference component
- **mag_alpha**: Weight for order of magnitude difference component
- **digit_alpha**: Weight for digit-wise EMD loss component
- **digit_exp**: Exponential factor for position-based weighting

Parameters for EMD:
- **tokenizer**: HuggingFace tokenizer for identifying digit tokens
- **batch_size**: Batch size for processing (to avoid number leakage between batches)
- **ntil_lambda**: Overall weight of EMD relative to MLE loss
- **digit_alpha**: Weight for digit-wise EMD loss component
- **digit_exp**: Exponential factor for position-based weighting

Minimal parameters:
- **tokenizer**: HuggingFace tokenizer for identifying digit tokens
- **ntil_lambda**: Overall weight of EMD relative to MLE loss

## Usage

Please refer to the `ntil.py` for usage.

If you want to use EMD loss only (faster training), please refer to the `emd.py`.


## Citation

If you use NTIL in your research, please cite:

```bibtex
@inproceedings{ntil2025,
  title={Advancing Sequential Numerical Prediction in Autoregressive Models},
  author={Fei, Xiang and Lu, Jinghui and Sun, Qi and Feng, Hao and Wang, Yanjie and Shi, Wei and Wang, An-Lan and Tang, Jingqun and Huang, Can},
  year={2025},
  booktitle={Proceedings of the 65rd Annual Meeting of the Association for Computational Linguistics (ACL)}
}
```