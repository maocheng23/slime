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

        For first response token comparison with SGLang prefill:
        - SGLang prefill outputs logits from position (prompt_len - 1)
        - These logits predict the token at position prompt_len
        - So we need hidden states at position (prompt_len - 1), not prompt_len

        FSDP layout: [batch, seq_len, hidden] or [seq_len, hidden]

        Saves hidden states at BOTH positions:
        - {name}: at position (prompt_len - 1) for prefill comparison
        - {name}_at_response_start: at position prompt_len for decode comparison
        """
        prompt_len = getattr(self, '_response_start_pos', 0)
        # For prefill comparison, we need position (prompt_len - 1)
        prefill_pos = max(0, prompt_len - 1)
        # For decode comparison, we need position prompt_len
        decode_pos = prompt_len

        def extract_token_at_position(t: torch.Tensor, pos: int) -> torch.Tensor:
            """Extract token at specified position from FSDP tensor layout."""
            if t.dim() == 3:
                # [batch, seq_len, hidden]
                seq_len = t.shape[1]
                if pos >= seq_len:
                    pos = seq_len - 1
                token = t[:, pos:pos+1, :]
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

        def save_at_position(t: torch.Tensor, pos: int, suffix: str = ""):
            key = f"{name}{suffix}"
            extracted = extract_token_at_position(t, pos).cpu()
            self._current_tensors[key] = extracted

        if isinstance(tensor, (tuple, list)):
            tensors = []
            for t in tensor:
                if t is not None and isinstance(t, torch.Tensor):
                    tensors.append(extract_token_at_position(t, prefill_pos).cpu())
            if len(tensors) == 1:
                self._current_tensors[name] = tensors[0]
            elif len(tensors) > 1:
                self._current_tensors[name] = tensors
        elif isinstance(tensor, torch.Tensor):
            # Save at prefill comparison position (prompt_len - 1)
            save_at_position(tensor, prefill_pos, "")
            # Also save at decode position (prompt_len) for reference
            if prefill_pos != decode_pos:
                save_at_position(tensor, decode_pos, "_at_response_start")

    def set_response_start_position(self, pos: int) -> None:
        """Set the response start position (prompt_len).

        This is used to determine extraction positions:
        - prefill comparison: position (pos - 1)
        - decode comparison: position pos
        """
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
        
        For true on-policy comparison:
        - SGLang decode at position N outputs logits predicting token at N+1
        - FSDP logits at position N also predict token at N+1
        - So we need to compare FSDP logits at position N with SGLang decode pass at position N
        
        Saves logits at ALL response positions:
        - logits_pos_X: logits at each response position (for decode comparison)
        - logits_at_prompt_end: at prompt_len - 1 (for prefill comparison)
        - logits_full: the complete logits tensor for all positions
        """
        prompt_len = getattr(self, '_response_start_pos', 0)
        
        if logits.dim() == 2:
            # [seq_len, vocab_size]
            seq_len = logits.shape[0]
            vocab_size = logits.shape[1]
            
            # Save sequence info
            self._current_tensors["seq_len"] = torch.tensor([seq_len])
            self._current_tensors["vocab_size"] = torch.tensor([vocab_size])
            self._current_tensors["prompt_len"] = torch.tensor([prompt_len])
            response_len = seq_len - prompt_len
            self._current_tensors["response_len"] = torch.tensor([response_len])
            
            # Save full logits tensor
            self._current_tensors["logits_full"] = logits.cpu().bfloat16()
            
            # Save logits at EACH response position for decode comparison
            response_positions = []
            for i in range(response_len):  # Save ALL response positions
                pos = prompt_len + i
                if pos < seq_len:
                    self._current_tensors[f"logits_pos_{pos}"] = logits[pos:pos+1, :].cpu().bfloat16()
                    response_positions.append(pos)
            
            self._current_tensors["response_logits_positions"] = torch.tensor(response_positions)
            
            # For first response token comparison:
            # SGLang prefill outputs logits predicting position prompt_len
            # FSDP logits at (prompt_len - 1) also predict position prompt_len
            if prompt_len > 0 and prompt_len - 1 < seq_len:
                self._current_tensors["logits_at_prompt_end"] = logits[prompt_len-1:prompt_len, :].cpu().bfloat16()
                self._current_tensors["logits_prompt_end_pos"] = torch.tensor([prompt_len - 1])
            
            # Also save the default position for backwards compatibility
            target_pos = prompt_len if prompt_len < seq_len else 0
            self._current_tensors["logits"] = logits[target_pos:target_pos+1, :].cpu().bfloat16()
            self._current_tensors["logits_pos"] = torch.tensor([target_pos])
            
            logger.info(f"[FSDPTensorDumper] Saved logits: seq_len={seq_len}, "
                       f"prompt_len={prompt_len}, response_len={response_len}, "
                       f"vocab_size={vocab_size}")
            
        elif logits.dim() == 3:
            # [batch, seq_len, vocab_size]
            batch_size = logits.shape[0]
            seq_len = logits.shape[1]
            vocab_size = logits.shape[2]
            
            # Save sequence info
            self._current_tensors["batch_size"] = torch.tensor([batch_size])
            self._current_tensors["seq_len"] = torch.tensor([seq_len])
            self._current_tensors["vocab_size"] = torch.tensor([vocab_size])
            self._current_tensors["prompt_len"] = torch.tensor([prompt_len])
            response_len = seq_len - prompt_len
            self._current_tensors["response_len"] = torch.tensor([response_len])
            
            # Save full logits tensor
            self._current_tensors["logits_full"] = logits.cpu().bfloat16()
            
            # Save logits at EACH response position
            response_positions = []
            for i in range(response_len):
                pos = prompt_len + i
                if pos < seq_len:
                    self._current_tensors[f"logits_pos_{pos}"] = logits[:, pos:pos+1, :].squeeze(1).cpu().bfloat16()
                    response_positions.append(pos)
            
            self._current_tensors["response_logits_positions"] = torch.tensor(response_positions)
            
            # Logits at prompt end
            if prompt_len > 0 and prompt_len - 1 < seq_len:
                self._current_tensors["logits_at_prompt_end"] = logits[:, prompt_len-1:prompt_len, :].squeeze(1).cpu().bfloat16()
                self._current_tensors["logits_prompt_end_pos"] = torch.tensor([prompt_len - 1])
            
            # Default position
            target_pos = prompt_len if prompt_len < seq_len else 0
            self._current_tensors["logits"] = logits[:, target_pos:target_pos+1, :].squeeze(1).cpu().bfloat16()
            self._current_tensors["logits_pos"] = torch.tensor([target_pos])
            
            logger.info(f"[FSDPTensorDumper] Saved logits: batch={batch_size}, seq_len={seq_len}, "
                       f"prompt_len={prompt_len}, response_len={response_len}, vocab_size={vocab_size}")
        else:
            self._current_tensors["logits"] = logits.cpu().bfloat16()

    def add_logprobs(self, logprobs: torch.Tensor) -> None:
        """Record log probabilities for debugging.

        IMPORTANT: log_probs from get_logprob_and_entropy_with_cp is SHIFTED:
        - log_probs has shape [total_seq_len - 1]
        - log_probs[i] = logprob of token at position (i + 1)

        For first response token at position prompt_len:
        - We need log_probs[prompt_len - 1]

        For comparison with SGLang prefill:
        - SGLang prefill outputs logits predicting position prompt_len
        - So we need the logprob for token at prompt_len
        - Which is log_probs[prompt_len - 1]
        """
        prompt_len = getattr(self, '_response_start_pos', 0)

        # For first response token: index = prompt_len - 1 (due to shift)
        first_response_idx = max(0, prompt_len - 1)

        # Save debug info
        self._current_tensors["logprobs_prompt_len"] = torch.tensor([prompt_len])
        self._current_tensors["logprobs_input_shape"] = torch.tensor(list(logprobs.shape))

        if logprobs.dim() == 1:
            # [seq_len - 1] (shifted)
            seq_len = logprobs.shape[0]
            self._current_tensors["logprobs_seq_len"] = torch.tensor([seq_len])

            # Check if logprobs is all zeros (bug indicator)
            is_all_zero = (logprobs.abs() < 1e-10).all().item()
            if is_all_zero:
                logger.warning(
                    f"[FSDPTensorDumper] WARNING: logprobs tensor is ALL ZEROS! "
                    f"shape={logprobs.shape}, prompt_len={prompt_len}"
                )
            else:
                non_zero_count = (logprobs.abs() > 1e-10).sum().item()
                logger.info(
                    f"[FSDPTensorDumper] logprobs: shape={logprobs.shape}, "
                    f"non_zero={non_zero_count}/{seq_len}, "
                    f"min={logprobs.min().item():.6f}, max={logprobs.max().item():.6f}"
                )

            # Save full logprobs for debugging
            self._current_tensors["logprobs_full"] = logprobs.cpu().bfloat16()

            # Extract first response token logprob
            if first_response_idx < seq_len:
                first_resp_lp = logprobs[first_response_idx]
                lp_val = first_resp_lp.item()
                self._current_tensors["logprobs"] = first_resp_lp.cpu().bfloat16().unsqueeze(0)
                self._current_tensors["logprobs_extracted_idx"] = torch.tensor(
                    [first_response_idx]
                )
                self._current_tensors["logprobs_extracted_value"] = torch.tensor([lp_val])

                if abs(lp_val) < 1e-10:
                    logger.warning(
                        f"[FSDPTensorDumper] WARNING: Extracted logprob is ZERO! "
                        f"idx={first_response_idx}, prompt_len={prompt_len}, "
                        f"seq_len={seq_len}"
                    )
                else:
                    logger.info(
                        f"[FSDPTensorDumper] Extracted first response logprob: "
                        f"idx={first_response_idx}, value={lp_val:.8f}"
                    )
            else:
                logger.warning(
                    f"[FSDPTensorDumper] first_response_idx={first_response_idx} "
                    f">= seq_len={seq_len}, using idx 0"
                )
                self._current_tensors["logprobs"] = logprobs[0:1].cpu().bfloat16()
                self._current_tensors["logprobs_extracted_idx"] = torch.tensor([0])

            # Also save first few response logprobs for comparison
            response_start_idx = first_response_idx
            response_end_idx = min(seq_len, first_response_idx + 5)
            if response_start_idx < seq_len:
                resp_logprobs = logprobs[response_start_idx:response_end_idx]
                self._current_tensors["response_logprobs_first5"] = (
                    resp_logprobs.cpu().bfloat16()
                )
                logger.info(
                    f"[FSDPTensorDumper] First 5 response logprobs "
                    f"(indices {response_start_idx}-{response_end_idx-1}): "
                    f"{resp_logprobs.tolist()}"
                )

        elif logprobs.dim() == 2:
            # [batch, seq_len - 1]
            seq_len = logprobs.shape[1]
            self._current_tensors["logprobs_seq_len"] = torch.tensor([seq_len])
            self._current_tensors["logprobs_full"] = logprobs.cpu().bfloat16()

            if first_response_idx < seq_len:
                first_resp_lp = logprobs[:, first_response_idx].cpu().bfloat16()
                self._current_tensors["logprobs"] = first_resp_lp
                self._current_tensors["logprobs_extracted_idx"] = torch.tensor(
                    [first_response_idx]
                )
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

