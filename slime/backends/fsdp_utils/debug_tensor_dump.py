"""
FSDP Tensor Dumper for layer-by-layer comparison with SGLang.

This module provides a mechanism to dump intermediate tensors from FSDP's
forward pass for comparison with SGLang's tensor dump.

Usage:
1. Set environment variables:
   - FSDP_TENSOR_DUMP_DIR: Output directory for tensor dumps
   - FSDP_TENSOR_DUMP_LAYERS: Comma-separated layer indices to dump (e.g., "0,1,2")

2. Call register_forward_hooks() on your model before running inference
3. Tensors will be saved to FSDP_TENSOR_DUMP_DIR

The naming convention mirrors SGLang's for easy comparison:
- Pass{pass_id:05d}.pt contains a dict of tensor_name -> tensor
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Global dumper instance
_global_dumper: Optional["FSDPTensorDumper"] = None


def get_fsdp_tensor_dumper() -> Optional["FSDPTensorDumper"]:
    """Get the global FSDP tensor dumper instance."""
    return _global_dumper


def set_fsdp_tensor_dumper(dumper: Optional["FSDPTensorDumper"]) -> None:
    """Set the global FSDP tensor dumper instance."""
    global _global_dumper
    _global_dumper = dumper


class FSDPTensorDumper:
    """Dumps intermediate tensors from FSDP forward passes for debugging."""

    def __init__(
        self,
        dump_dir: str,
        dump_layers: Optional[list[int]] = None,
        rank: int = 0,
    ):
        self._dump_dir = Path(dump_dir)
        self._dump_layers = dump_layers
        self._forward_pass_id = 0
        self._current_tensors: dict[str, torch.Tensor] = {}
        self._pid = os.getpid()

        self._process_dir = self._dump_dir / f"Rank{rank}_pid{self._pid}"
        self._process_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[FSDPTensorDumper] Initialized. dump_dir={self._process_dir}, dump_layers={dump_layers}"
        )

    def add_tensor(self, name: str, tensor: torch.Tensor | tuple | list) -> None:
        """Add a tensor to the current forward pass collection.
        
        Saves the FIRST RESPONSE token position to match SGLang's decode phase.
        FSDP layout: [batch, seq_len, hidden] or [seq_len, hidden]
        
        If response_start_position is set, extract from that position.
        Otherwise, extract from position 0 as fallback.
        """
        target_pos = getattr(self, '_response_start_pos', 0)
        
        def extract_token_at_position(t: torch.Tensor, pos: int) -> torch.Tensor:
            """Extract token at specified position from FSDP tensor layout."""
            if t.dim() == 3:
                # [batch, seq_len, hidden]
                seq_len = t.shape[1]
                if pos >= seq_len:
                    pos = seq_len - 1  # Use last token if pos is out of bounds
                token = t[:, pos:pos+1, :]  # Keep dim: [batch, 1, hidden]
                if token.shape[0] == 1:
                    return token[0, :, :]  # [1, hidden]
                return token[:, 0, :]  # [batch, hidden]
            elif t.dim() == 2:
                # [seq_len, hidden]
                seq_len = t.shape[0]
                if pos >= seq_len:
                    pos = seq_len - 1
                return t[pos:pos+1, :]  # [1, hidden]
            else:
                return t
        
        if isinstance(tensor, (tuple, list)):
            tensors = []
            for t in tensor:
                if t is not None and isinstance(t, torch.Tensor):
                    tensors.append(extract_token_at_position(t, target_pos).cpu())
            if len(tensors) == 1:
                self._current_tensors[name] = tensors[0]
            elif len(tensors) > 1:
                self._current_tensors[name] = tensors
        elif isinstance(tensor, torch.Tensor):
            self._current_tensors[name] = extract_token_at_position(tensor, target_pos).cpu()

    def set_response_start_position(self, pos: int) -> None:
        """Set the response start position for token extraction."""
        self._response_start_pos = pos
        logger.info(f"[FSDPTensorDumper] Set response_start_pos={pos}")

    def dump_current_tensors(self) -> None:
        """Dump all collected tensors for the current forward pass."""
        if len(self._current_tensors) == 0:
            return

        tensor_file = self._process_dir / f"Pass{self._forward_pass_id:05d}.pt"
        logger.info(
            f"[FSDPTensorDumper] Dumping pass {self._forward_pass_id} "
            f"with {len(self._current_tensors)} tensors to {tensor_file}"
        )
        torch.save(self._current_tensors, str(tensor_file))
        self._current_tensors = {}
        self._forward_pass_id += 1

    def should_dump_layer(self, layer_idx: int) -> bool:
        """Check if a layer should be dumped."""
        if self._dump_layers is None:
            return True
        return layer_idx in self._dump_layers

    def register_forward_hooks(self, model: torch.nn.Module) -> None:
        """
        Register forward hooks on the model to capture intermediate tensors.
        
        This hooks into the transformer layers to capture:
        - Embedding output
        - Input hidden states (before each layer)
        - Output hidden states (after each layer)
        - Attention output (after self-attention)
        - MLP output (after MLP)
        - Final output (lm_head)
        """
        hooks_registered = 0

        # Hook embedding layer
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embedding = model.model.embed_tokens
            embedding.register_forward_hook(self._create_named_hook("model.embed_tokens"))
            hooks_registered += 1
            logger.info("[FSDPTensorDumper] Hooked embedding layer")

        # Hook output layer (lm_head)
        if hasattr(model, 'lm_head'):
            output_layer = model.lm_head
            output_layer.register_forward_hook(self._create_named_hook("lm_head"))
            hooks_registered += 1
            logger.info("[FSDPTensorDumper] Hooked output_layer (lm_head)")

        # Find the transformer layers
        layers = self._find_transformer_layers(model)

        for layer_idx, layer in enumerate(layers):
            if not self.should_dump_layer(layer_idx):
                continue

            # Hook for layer input/output
            layer.register_forward_hook(self._create_layer_hook(layer_idx))
            hooks_registered += 1

            # Try to hook into attention and MLP sublayers
            self._hook_sublayers(layer, layer_idx)

        logger.info(
            f"[FSDPTensorDumper] Registered {hooks_registered} hooks on {len(layers)} layers"
        )

    def _find_transformer_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """Find transformer layers in the model."""
        layers = []

        # Try to find model.layers (HuggingFace structure)
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = list(model.model.layers)
            logger.info(f"[FSDPTensorDumper] Found model.model.layers with {len(layers)} layers")
            return layers

        # Try decoder.layers
        if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            layers = list(model.model.decoder.layers)
            logger.info(f"[FSDPTensorDumper] Found decoder.layers with {len(layers)} layers")
            return layers

        # Fallback: search for ModuleList containing layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList):
                # Check if this looks like a transformer layer list
                if len(module) > 0 and hasattr(module[0], 'self_attn'):
                    layers = list(module)
                    logger.info(f"[FSDPTensorDumper] Found {name} with {len(layers)} layers")
                    return layers

        logger.warning("[FSDPTensorDumper] Could not find transformer layers!")
        return layers

    def _create_named_hook(self, name: str):
        """Create a forward hook for a named module."""

        def hook(module, input_tuple, output):
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output

            if isinstance(out_tensor, torch.Tensor):
                self.add_tensor(name, out_tensor.bfloat16())

        return hook

    def _create_layer_hook(self, layer_idx: int):
        """Create a forward hook for a transformer layer."""

        def hook(module, input_tuple, output):
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output

            if isinstance(out_tensor, torch.Tensor):
                self.add_tensor(f"layer_{layer_idx}_output", out_tensor.bfloat16())

        return hook

    def _hook_sublayers(self, layer: torch.nn.Module, layer_idx: int) -> None:
        """Hook into attention and MLP sublayers."""
        # Hook input_layernorm output (to match Megatron naming)
        if hasattr(layer, 'input_layernorm'):
            layer.input_layernorm.register_forward_hook(
                self._create_sublayer_hook(layer_idx, "input_layernorm")
            )
        
        # Hook post_attention_layernorm output (to match Megatron naming)
        if hasattr(layer, 'post_attention_layernorm'):
            layer.post_attention_layernorm.register_forward_hook(
                self._create_sublayer_hook(layer_idx, "post_attention_layernorm")
            )
        
        # Hook self-attention
        if hasattr(layer, 'self_attn'):
            layer.self_attn.register_forward_hook(
                self._create_sublayer_hook(layer_idx, "self_attention")
            )

        # Hook MLP
        if hasattr(layer, 'mlp'):
            layer.mlp.register_forward_hook(
                self._create_sublayer_hook(layer_idx, "mlp")
            )
            # Hook MLP sublayers to match Megatron naming
            if hasattr(layer.mlp, 'gate_proj') and hasattr(layer.mlp, 'up_proj'):
                # For models with separate gate/up projections
                layer.mlp.gate_proj.register_forward_hook(
                    self._create_sublayer_hook(layer_idx, "mlp.gate_up_proj")
                )
                layer.mlp.up_proj.register_forward_hook(
                    self._create_sublayer_hook(layer_idx, "mlp.gate_up_proj")
                )
            if hasattr(layer.mlp, 'down_proj'):
                layer.mlp.down_proj.register_forward_hook(
                    self._create_sublayer_hook(layer_idx, "mlp.down_proj")
                )

    def _create_sublayer_hook(self, layer_idx: int, sublayer_name: str):
        """Create a forward hook for a sublayer (attention or MLP).
        
        Naming convention matches Megatron:
        - layer_{idx}_input_layernorm_output
        - layer_{idx}_post_attention_layernorm_output
        - layer_{idx}_self_attention_output
        - layer_{idx}_mlp_output
        - layer_{idx}_mlp.gate_up_proj_output
        - layer_{idx}_mlp.down_proj_output
        """

        def hook(module, input_tuple, output):
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output

            if isinstance(out_tensor, torch.Tensor):
                # Use Megatron naming convention
                if "." in sublayer_name:
                    # For nested names like "mlp.gate_up_proj"
                    tensor_name = f"layer_{layer_idx}_{sublayer_name}_output"
                else:
                    tensor_name = f"layer_{layer_idx}_{sublayer_name}_output"
                self.add_tensor(tensor_name, out_tensor.bfloat16())

        return hook

    def add_input_ids(self, input_ids: torch.Tensor) -> None:
        """Record the input token IDs for debugging."""
        # Save full input_ids for reference
        self._current_tensors["fsdp_input_ids"] = input_ids.cpu()
        
        # Save the token at the response start position (what we're actually comparing)
        target_pos = getattr(self, '_response_start_pos', 0)
        flat_ids = input_ids.flatten()
        if target_pos < len(flat_ids):
            self._current_tensors["fsdp_compared_token_id"] = flat_ids[target_pos:target_pos+1].cpu()
            self._current_tensors["fsdp_compared_position"] = torch.tensor([target_pos])
        else:
            # Fallback to first token
            self._current_tensors["fsdp_compared_token_id"] = flat_ids[0:1].cpu()
            self._current_tensors["fsdp_compared_position"] = torch.tensor([0])

    def add_logits(self, logits: torch.Tensor) -> None:
        """Record logits for debugging.
        
        Saves logits at multiple positions for comparison:
        - logits: at response_start_pos (for response token comparison)
        - logits_at_prompt_end: at prompt_len - 1 (for prefill comparison with SGLang)
        - logits_last: at the last position (alternative for prefill)
        """
        target_pos = getattr(self, '_response_start_pos', 0)
        
        if logits.dim() == 2:
            # [seq_len, vocab_size]
            seq_len = logits.shape[0]
            # Logits at response start position
            if target_pos < seq_len:
                self._current_tensors["logits"] = logits[target_pos:target_pos+1, :].cpu().bfloat16()
            else:
                self._current_tensors["logits"] = logits[-1:, :].cpu().bfloat16()
            
            # Logits at prompt end (last prompt position = target_pos - 1)
            # This is what predicts the first response token
            prompt_end_pos = max(0, target_pos - 1) if target_pos > 0 else seq_len - 1
            self._current_tensors["logits_at_prompt_end"] = logits[prompt_end_pos:prompt_end_pos+1, :].cpu().bfloat16()
            self._current_tensors["logits_prompt_end_pos"] = torch.tensor([prompt_end_pos])
            
            # Also save the very last position (for SGLang prefill matching)
            self._current_tensors["logits_last"] = logits[-1:, :].cpu().bfloat16()
            self._current_tensors["logits_last_pos"] = torch.tensor([seq_len - 1])
            
        elif logits.dim() == 3:
            # [batch, seq_len, vocab_size]
            seq_len = logits.shape[1]
            if target_pos < seq_len:
                self._current_tensors["logits"] = logits[:, target_pos:target_pos+1, :].squeeze(1).cpu().bfloat16()
            else:
                self._current_tensors["logits"] = logits[:, -1:, :].squeeze(1).cpu().bfloat16()
            
            # Logits at prompt end
            prompt_end_pos = max(0, target_pos - 1) if target_pos > 0 else seq_len - 1
            self._current_tensors["logits_at_prompt_end"] = logits[:, prompt_end_pos:prompt_end_pos+1, :].squeeze(1).cpu().bfloat16()
            self._current_tensors["logits_prompt_end_pos"] = torch.tensor([prompt_end_pos])
            
            # Last position
            self._current_tensors["logits_last"] = logits[:, -1:, :].squeeze(1).cpu().bfloat16()
            self._current_tensors["logits_last_pos"] = torch.tensor([seq_len - 1])
        else:
            self._current_tensors["logits"] = logits.cpu().bfloat16()

    def add_logprobs(self, logprobs: torch.Tensor) -> None:
        """Record log probabilities for debugging."""
        # Extract logprobs at response start position
        target_pos = getattr(self, '_response_start_pos', 0)
        if logprobs.dim() == 1:
            # [seq_len]
            if target_pos < logprobs.shape[0]:
                self._current_tensors["logprobs"] = logprobs[target_pos:target_pos+1].cpu().bfloat16()
            else:
                self._current_tensors["logprobs"] = logprobs[0:1].cpu().bfloat16()
        elif logprobs.dim() == 2:
            # [batch, seq_len]
            if target_pos < logprobs.shape[1]:
                self._current_tensors["logprobs"] = logprobs[:, target_pos:target_pos+1].cpu().bfloat16()
            else:
                self._current_tensors["logprobs"] = logprobs[:, 0:1].cpu().bfloat16()
        else:
            self._current_tensors["logprobs"] = logprobs.cpu().bfloat16()


def register_fsdp_tensor_hooks(model: torch.nn.Module) -> None:
    """Register tensor dump hooks on FSDP model."""
    dump_dir = os.environ.get("FSDP_TENSOR_DUMP_DIR", "")
    dump_layers_str = os.environ.get("FSDP_TENSOR_DUMP_LAYERS", "")
    
    if not dump_dir:
        return
    
    dump_layers = None
    if dump_layers_str:
        dump_layers = [int(x.strip()) for x in dump_layers_str.split(",")]
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    dumper = FSDPTensorDumper(
        dump_dir=dump_dir,
        dump_layers=dump_layers,
        rank=rank,
    )
    
    dumper.register_forward_hooks(model)
    set_fsdp_tensor_dumper(dumper)
    
    logger.info(f"[FSDPTensorDumper] Registered tensor dump hooks for model")

