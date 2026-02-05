# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Retrieves the pretrained models for Moshi and Mimi."""
from pathlib import Path
import logging
from safetensors.torch import load_model, load_file
import torch
from ..utils.logging import setup_logger
logger = setup_logger(__name__)
from .compression import MimiModel
from .lm import LMModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..quantization import SplitResidualVectorQuantizer

SAMPLE_RATE = 24000
FRAME_RATE = 12.5
TEXT_TOKENIZER_NAME = 'tokenizer_spm_32k_3.model'
MOSHI_NAME = 'model.safetensors'
MIMI_NAME = 'tokenizer-e351c8d8-checkpoint125.safetensors'
DEFAULT_REPO = 'nvidia/personaplex-7b-v1'

_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}
_lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": _quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}

def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def _calculate_model_size(model: torch.nn.Module) -> dict:
    """Calculate detailed model size statistics."""
    total_params = 0
    total_bytes = 0
    param_details = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        num_bytes = param.numel() * param.element_size()
        total_params += num_params
        total_bytes += num_bytes
        param_details[name] = {
            'shape': tuple(param.shape),
            'dtype': str(param.dtype),
            'num_params': num_params,
            'size_mb': num_bytes / (1024 ** 2)
        }
    
    return {
        'total_params': total_params,
        'total_size_mb': total_bytes / (1024 ** 2),
        'total_size_gb': total_bytes / (1024 ** 3),
        'param_details': param_details
    }


def _log_model_statistics(model: torch.nn.Module, stage: str = ""):
    """Log comprehensive model statistics."""
    stats = _calculate_model_size(model)
    
    logger.info(f"{'='*80}")
    logger.info(f"Model Statistics {stage}")
    logger.info(f"{'='*80}")
    logger.info(f"Total Parameters: {stats['total_params']:,}")
    logger.info(f"Total Size: {stats['total_size_mb']:.2f} MB ({stats['total_size_gb']:.4f} GB)")
    logger.info(f"Number of parameter tensors: {len(stats['param_details'])}")
    
    # Group by module prefix
    module_stats = {}
    for name, details in stats['param_details'].items():
        prefix = name.split('.')[0] if '.' in name else name
        if prefix not in module_stats:
            module_stats[prefix] = {'params': 0, 'size_mb': 0}
        module_stats[prefix]['params'] += details['num_params']
        module_stats[prefix]['size_mb'] += details['size_mb']
    
    logger.info(f"\nSize by module:")
    for module, stats_dict in sorted(module_stats.items(), key=lambda x: x[1]['size_mb'], reverse=True):
        logger.info(f"  {module:30s}: {stats_dict['params']:12,} params, {stats_dict['size_mb']:8.2f} MB")
    logger.info(f"{'='*80}\n")
    
    return stats


def _compare_state_dicts(model_sd: dict, loaded_sd: dict, stage: str = ""):
    """Compare model state dict with loaded state dict."""
    logger.info(f"{'='*80}")
    logger.info(f"State Dict Comparison {stage}")
    logger.info(f"{'='*80}")
    
    model_keys = set(model_sd.keys())
    loaded_keys = set(loaded_sd.keys())
    
    missing_keys = model_keys - loaded_keys
    extra_keys = loaded_keys - model_keys
    common_keys = model_keys & loaded_keys
    
    logger.info(f"Model state dict keys: {len(model_keys)}")
    logger.info(f"Loaded state dict keys: {len(loaded_keys)}")
    logger.info(f"Common keys: {len(common_keys)}")
    logger.info(f"Missing keys (in model, not in checkpoint): {len(missing_keys)}")
    logger.info(f"Extra keys (in checkpoint, not in model): {len(extra_keys)}")
    
    if missing_keys:
        logger.warning(f"\nMissing keys ({len(missing_keys)}):")
        for key in sorted(list(missing_keys)[:20]):  # Show first 20
            if key in model_sd:
                logger.warning(f"  {key}: shape={model_sd[key].shape}, dtype={model_sd[key].dtype}")
        if len(missing_keys) > 20:
            logger.warning(f"  ... and {len(missing_keys) - 20} more")
    
    if extra_keys:
        logger.warning(f"\nExtra keys ({len(extra_keys)}):")
        for key in sorted(list(extra_keys)[:20]):  # Show first 20
            if key in loaded_sd:
                tensor = loaded_sd[key]
                logger.warning(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        if len(extra_keys) > 20:
            logger.warning(f"  ... and {len(extra_keys) - 20} more")
    
    # Check shape mismatches for common keys
    shape_mismatches = []
    for key in common_keys:
        if model_sd[key].shape != loaded_sd[key].shape:
            shape_mismatches.append({
                'key': key,
                'model_shape': model_sd[key].shape,
                'loaded_shape': loaded_sd[key].shape
            })
    
    if shape_mismatches:
        logger.warning(f"\nShape mismatches ({len(shape_mismatches)}):")
        for mismatch in shape_mismatches[:20]:  # Show first 20
            logger.warning(f"  {mismatch['key']}:")
            logger.warning(f"    Model:  {mismatch['model_shape']}")
            logger.warning(f"    Loaded: {mismatch['loaded_shape']}")
        if len(shape_mismatches) > 20:
            logger.warning(f"  ... and {len(shape_mismatches) - 20} more")
    
    logger.info(f"{'='*80}\n")
    
    return {
        'missing_keys': missing_keys,
        'extra_keys': extra_keys,
        'shape_mismatches': shape_mismatches
    }


def get_mimi(filename: str | Path,
             device: torch.device | str = 'cpu') -> MimiModel:
    """Return a pretrained Mimi model."""
    logger.info(f"Loading Mimi model from: {filename}")
    logger.info(f"Target device: {device}")
    
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    
    logger.info("Mimi model architecture created")
    _log_model_statistics(model, "- After Architecture Creation")
    
    model.eval()
    if _is_safetensors(filename):
        logger.info("Loading weights using safetensors")
        load_model(model, filename)
    else:
        logger.info("Loading weights using torch.load")
        pkg = torch.load(filename, "cpu")
        model.load_state_dict(pkg["model"])
    
    _log_model_statistics(model, "- After Weight Loading")
    
    model.set_num_codebooks(8)
    logger.info("Mimi model loaded successfully\n")
    return model


def get_moshi_lm(
    filename: str | Path | None,
    copy_missing_weights: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    delays=None,
    cpu_offload: bool = False,
) -> LMModel:
    """Return a pretrained Moshi LM model.
    Args:
        filename: Path to model weights.
        copy_missing_weights: Whether to copy missing weights from existing layers.
        device: Target device for the model.
        dtype: Data type for model weights.
        delays: Optional custom delays configuration.
        cpu_offload: If True, offload model layers to CPU when GPU memory is
                     insufficient. Uses accelerate's device_map="auto".
    """
    logger.info(f"{'#'*80}")
    logger.info(f"Loading Moshi LM Model")
    logger.info(f"{'#'*80}")
    logger.info(f"Filename: {filename}")
    logger.info(f"Copy missing weights: {copy_missing_weights}")
    logger.info(f"Target device: {device}")
    logger.info(f"Target dtype: {dtype}")
    logger.info(f"CPU offload: {cpu_offload}")
    
    # Copy to avoid mutating a shared/global dict
    lm_kwargs = dict(_lm_kwargs)
    
    # IMPORTANT: Check if we should modify dep_q based on checkpoint
    # Original model has dep_q=8, but some checkpoints may have different values
    if filename is not None:
        logger.info(f"Original dep_q in config: {lm_kwargs['dep_q']}")
        # We'll inspect the checkpoint to determine the actual dep_q
        # For now, keeping the modification but adding a warning
        lm_kwargs["dep_q"] = 16
        logger.warning(f"Modified dep_q from 8 to 16 - this may cause issues if checkpoint was trained with dep_q=8")
        logger.warning(f"Missing embeddings will be copied from existing ones, which may affect audio quality")
    else:
        lm_kwargs["dep_q"] = 16
        logger.info(f"Modified dep_q from 8 to 16")
    
    if delays is not None:
        lm_kwargs["delays"] = delays
        logger.info(f"Using custom delays: {delays}")
    
    if cpu_offload and filename is not None:
        return _get_moshi_lm_with_offload(
            filename, copy_missing_weights, device, dtype, lm_kwargs
        )
    
    # Init with meta device to avoid init dummy memory
    init_device = "meta" if filename is not None else device
    logger.info(f"Initializing model on device: {init_device}")
    
    model = LMModel(device=init_device, dtype=dtype, **lm_kwargs)
    
    if filename is None:
        logger.info("No filename provided, returning uninitialized model")
        model.to(device=device, dtype=dtype)
        model.eval()
        _log_model_statistics(model, "- Uninitialized Model")
        return model
    
    logger.info("Model architecture created")
    if init_device != "meta":
        _log_model_statistics(model, "- After Architecture Creation (before loading)")
    
    filename = str(filename)
    
    # Load state_dict
    logger.info(f"Loading checkpoint from: {filename}")
    if filename.endswith(".safetensors"):
        dev = torch.device(device) if isinstance(device, str) else device
        load_device = "cpu" if dev.type == "mps" else dev.type
        logger.info(f"Loading safetensors to device: {load_device}")
        state_dict = load_file(filename, device=load_device)
    else:
        logger.info("Loading torch checkpoint to CPU")
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
    
    logger.info(f"Checkpoint loaded with {len(state_dict)} keys")
    
    # Calculate checkpoint size
    checkpoint_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    logger.info(f"Checkpoint total size: {checkpoint_size / (1024**2):.2f} MB ({checkpoint_size / (1024**3):.4f} GB)")
    
    # Detect actual dep_q from checkpoint
    depformer_emb_keys = [k for k in state_dict.keys() if k.startswith('depformer_emb.') and k.endswith('.weight')]
    if depformer_emb_keys:
        # Extract indices from keys like "depformer_emb.0.weight"
        indices = []
        for key in depformer_emb_keys:
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                indices.append(int(parts[1]))
        actual_dep_q_in_checkpoint = max(indices) + 1 if indices else None
        logger.info(f"Detected dep_q in checkpoint: {actual_dep_q_in_checkpoint} (indices: {sorted(indices)})")
        logger.info(f"Model expects dep_q: {lm_kwargs['dep_q']}")
        
        if actual_dep_q_in_checkpoint and actual_dep_q_in_checkpoint < lm_kwargs['dep_q']:
            logger.warning(f"⚠️  MISMATCH: Checkpoint has dep_q={actual_dep_q_in_checkpoint} but model expects {lm_kwargs['dep_q']}")
            logger.warning(f"⚠️  Missing {lm_kwargs['dep_q'] - actual_dep_q_in_checkpoint} embeddings will be copied from existing ones")
            logger.warning(f"⚠️  This may affect model performance and audio quality!")
    
    # Get model state dict for comparison
    model_sd = model.state_dict()
    logger.info(f"Model state dict contains {len(model_sd)} keys")
    
    # Compare before patching
    comparison = _compare_state_dicts(model_sd, state_dict, "- Before Patching")
    
    # Patch 1: expand depformer self_attn weights if needed
    logger.info("\n--- Patch 1: Expanding depformer self_attn weights ---")
    patch1_count = 0
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                logger.info(f"Expanding {name}")
                logger.info(f"  Original shape: {tensor.shape}")
                logger.info(f"  Target shape: {model_sd[name].shape}")
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0] :]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)
                logger.info(f"  New shape: {state_dict[name].shape}")
                patch1_count += 1
    logger.info(f"Patch 1 complete: expanded {patch1_count} tensors")
    
    # Patch 2: fill missing keys by copying 0..7 -> 8..15 for certain groups
    logger.info("\n--- Patch 2: Copying weights for missing keys ---")
    patch2_count = 0
    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            logger.info(f"Replacing {name} <- {src}")
                            logger.info(f"  Shape: {state_dict[src].shape}, dtype: {state_dict[src].dtype}")
                            state_dict[name] = state_dict[src]
                            replaced = True
                            patch2_count += 1
                        break
                if replaced:
                    break
            
            # If still not replaced, try to find ANY available source from index 0-7
            if not replaced and name not in state_dict:
                for rep in to_replace:
                    # Extract the pattern like "depformer_emb.7.weight"
                    if f"{rep}." in name:
                        # Try to find any available source with index 0-7
                        for fallback_idx in range(8):
                            # Extract the base pattern by replacing the index
                            import re
                            pattern = rf"{rep}\.(\d+)\."
                            match = re.search(pattern, name)
                            if match:
                                fallback_src = name.replace(f"{rep}.{match.group(1)}.", f"{rep}.{fallback_idx}.")
                                if fallback_src in state_dict:
                                    logger.info(f"Replacing {name} <- {fallback_src} (fallback)")
                                    logger.info(f"  Shape: {state_dict[fallback_src].shape}, dtype: {state_dict[fallback_src].dtype}")
                                    state_dict[name] = state_dict[fallback_src]
                                    replaced = True
                                    patch2_count += 1
                                    break
                        if replaced:
                            break
            
            if not replaced and name not in state_dict:
                logger.warning(f"Missing {name} (shape: {model_sd[name].shape})")
    
    logger.info(f"Patch 2 complete: copied {patch2_count} tensors")
    
    # Compare after patching
    _compare_state_dicts(model_sd, state_dict, "- After Patching")
    
    # Assign weights to target device
    dev = torch.device(device) if isinstance(device, str) else device
    logger.info(f"\nMoving weights to device: {dev}, dtype: {dtype}")
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device=dev, dtype=dtype)
    
    logger.info("Loading state dict into model...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    
    if missing:
        logger.warning(f"Missing keys after load_state_dict: {len(missing)}")
        for key in missing[:10]:
            logger.warning(f"  {key}")
        if len(missing) > 10:
            logger.warning(f"  ... and {len(missing) - 10} more")
    
    if unexpected:
        logger.warning(f"Unexpected keys after load_state_dict: {len(unexpected)}")
        for key in unexpected[:10]:
            logger.warning(f"  {key}")
        if len(unexpected) > 10:
            logger.warning(f"  ... and {len(unexpected) - 10} more")
    
    model.eval()
    final_model = model.to(device=device, dtype=dtype)
    
    _log_model_statistics(final_model, "- Final Model (after loading)")
    logger.info(f"{'#'*80}")
    logger.info("Moshi LM model loading complete")
    logger.info(f"{'#'*80}\n")
    
    return final_model


def _get_moshi_lm_with_offload(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    lm_kwargs: dict,
) -> LMModel:
    """Load Moshi LM with CPU offloading using accelerate.
    This function distributes model layers across GPU and CPU based on
    available GPU memory. Layers that don't fit on GPU are kept on CPU
    and moved to GPU only during forward pass.
    """
    try:
        from accelerate import infer_auto_device_map, dispatch_model
    except ImportError:
        raise ImportError(
            "CPU offloading requires the 'accelerate' package. "
            "Install it with: pip install accelerate"
        )
    
    filename = str(filename)
    logger.info("="*80)
    logger.info("Loading model with CPU offloading enabled")
    logger.info("="*80)
    
    # First, create model on CPU to get the architecture
    logger.info("Creating model architecture on CPU...")
    model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)
    _log_model_statistics(model, "- Initial Model on CPU")
    
    # Load state_dict to CPU
    logger.info(f"Loading checkpoint from: {filename}")
    if filename.endswith(".safetensors"):
        state_dict = load_file(filename, device="cpu")
    else:
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
    
    logger.info(f"Checkpoint loaded with {len(state_dict)} keys")
    checkpoint_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    logger.info(f"Checkpoint size: {checkpoint_size / (1024**2):.2f} MB")
    
    # Get model state dict for comparison
    model_sd = model.state_dict()
    _compare_state_dicts(model_sd, state_dict, "- Before Patching (Offload Mode)")
    
    # Apply weight patches (same as non-offload path)
    logger.info("\nApplying weight patches...")
    patch_count = 0
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                logger.info(f"Expanding {name}: {tensor.shape} -> {model_sd[name].shape}")
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0]:]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)
                patch_count += 1
    
    logger.info(f"Expanded {patch_count} tensors")
    
    copy_count = 0
    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            logger.info(f"Replacing {name} <- {src}")
                            state_dict[name] = state_dict[src]
                            replaced = True
                            copy_count += 1
                        break
                if replaced:
                    break
            if not replaced:
                logger.warning(f"Missing {name}")
    
    logger.info(f"Copied {copy_count} tensors for missing keys")
    
    logger.info("\nLoading state dict into model...")
    model.load_state_dict(state_dict, strict=False, assign=True)
    
    # Determine target device
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type != "cuda":
        logger.info(f"CPU offload requested but device is {dev}, skipping offload")
        model.to(dev)
        model.eval()
        _log_model_statistics(model, f"- Final Model on {dev}")
        return model
    
    # Infer device map based on available GPU memory
    logger.info("\nInferring device map for GPU/CPU distribution...")
    device_map = infer_auto_device_map(
        model,
        max_memory=None,
        no_split_module_classes=["StreamingTransformerLayer"],
        dtype=dtype,
    )
    
    # Log the device distribution
    gpu_modules = [k for k, v in device_map.items() if v == 0 or v == "cuda:0"]
    cpu_modules = [k for k, v in device_map.items() if v == "cpu"]
    
    logger.info(f"\nDevice map computed:")
    logger.info(f"  GPU modules: {len(gpu_modules)}")
    logger.info(f"  CPU modules: {len(cpu_modules)}")
    
    if gpu_modules:
        logger.info(f"\nFirst 10 GPU modules:")
        for module in gpu_modules[:10]:
            logger.info(f"    {module}")
        if len(gpu_modules) > 10:
            logger.info(f"    ... and {len(gpu_modules) - 10} more")
    
    if cpu_modules:
        logger.info(f"\nFirst 10 CPU modules:")
        for module in cpu_modules[:10]:
            logger.info(f"    {module}")
        if len(cpu_modules) > 10:
            logger.info(f"    ... and {len(cpu_modules) - 10} more")
    
    # Dispatch model across devices
    logger.info("\nDispatching model across devices...")
    model = dispatch_model(
        model,
        device_map=device_map,
        offload_dir="offload_weights",
    )
    
    model.eval()
    logger.info("Model dispatched successfully")
    logger.info("="*80)
    
    return model