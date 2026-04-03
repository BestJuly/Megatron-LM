#!/usr/bin/env python3
"""Dump TransformerConfig from both Bridge and MIMO for comparison."""
import json
import os
import sys
import dataclasses

import torch

def config_to_dict(cfg):
    """Convert a dataclass config to a comparable dict."""
    d = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if callable(v) and not isinstance(v, (type, torch.dtype)):
            d[f.name] = str(v)
        elif isinstance(v, torch.dtype):
            d[f.name] = str(v)
        elif isinstance(v, list):
            d[f.name] = v
        elif dataclasses.is_dataclass(v):
            d[f.name] = str(type(v).__name__)
        else:
            try:
                json.dumps(v)
                d[f.name] = v
            except (TypeError, ValueError):
                d[f.name] = str(v)
    return d

def dump_bridge_config():
    """Build config via Bridge AutoBridge path."""
    sys.path.insert(0, "/lustre/fs1/portfolios/coreai/users/jinliangl/repos/Megatron-Bridge/src")
    sys.path.insert(0, "/lustre/fs1/portfolios/coreai/users/jinliangl/repos/Megatron-Bridge/3rdparty/Megatron-LM")

    from megatron.bridge import AutoBridge
    bridge = AutoBridge.from_hf_pretrained("moonshotai/Kimi-K2.5", trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)

    # Proxy overrides matching checkpoint
    provider.hidden_size = 7168
    provider.ffn_hidden_size = 1024
    provider.num_moe_experts = 16
    provider.moe_ffn_hidden_size = 64
    provider.num_layers = 4
    provider.seq_length = 2048
    provider.moe_layer_freq = [0, 1, 1, 1]

    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 8
    provider.sequence_parallel = False
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16

    provider.finalize()
    return config_to_dict(provider)

def dump_mimo_config():
    """Build config via MIMO path."""
    sys.path.insert(0, "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_mcore/users/jinliangl/repos/Megatron-LM")
    sys.path.insert(0, "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_mcore/users/jinliangl/repos/Megatron-LM/examples/mimo")

    from model_providers.kimi_k25 import get_kimi_k25_language_config
    cfg = get_kimi_k25_language_config(variant="proxy")
    cfg.tensor_model_parallel_size = 1
    cfg.pipeline_model_parallel_size = 1
    cfg.expert_model_parallel_size = 8
    cfg.sequence_parallel = False
    cfg.bf16 = True
    return config_to_dict(cfg)

if __name__ == "__main__":
    bridge = dump_bridge_config()
    mimo = dump_mimo_config()

    all_keys = sorted(set(bridge.keys()) | set(mimo.keys()))
    diffs = []
    for k in all_keys:
        bv = bridge.get(k, "MISSING")
        mv = mimo.get(k, "MISSING")
        if bv != mv:
            diffs.append((k, bv, mv))

    print(f"Total fields: {len(all_keys)}")
    print(f"Differences:  {len(diffs)}")
    print()
    for k, bv, mv in diffs:
        print(f"  {k}:")
        print(f"    Bridge: {bv}")
        print(f"    MIMO:   {mv}")
        print()
