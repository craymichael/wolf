__author__ = 'max'

import torch
import torch.nn as nn


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    # https://github.com/huggingface/transformers/issues/9377#issuecomment-753504373
    # if not export and torch.cuda.is_available():
    #     try:
    #         from apex.normalization import FusedLayerNorm
    #         return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    #     except ImportError:
    #         pass
    return nn.LayerNorm(normalized_shape, eps, elementwise_affine)
