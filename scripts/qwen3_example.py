#!/usr/bin/env python3
"""Offline random Qwen3 forward and optional local HF export, without a Tokenizer.

Usage: .venv/bin/python scripts/qwen3_example.py model.num_hidden_layers=2 ...
       .venv/bin/python scripts/qwen3_example.py --export outputs/qwen3/hf
"""
import argparse
import json
from pathlib import Path

import torch

from mahjong.app.assembly import build_model
from mahjong.app.config import parse_model
from mahjong.contracts import Batch
from mahjong.models.qwen3 import INPUT_SCHEMA, load_qwen3_local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path)
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = parse_model(args.overrides)
    adapter = build_model(config).eval()
    native = adapter.causal_lm
    ids = torch.randint(0, native.config.vocab_size, (2, 8),
                        generator=torch.Generator().manual_seed(20261012))
    with torch.no_grad():
        output = native(input_ids=ids, labels=ids)
        result = adapter(Batch({'input_ids': ids, 'labels': ids.clone()},
                               ('example-0', 'example-1'), 2, INPUT_SCHEMA), phase='eval')
    print(json.dumps({'parameters': sum(p.numel() for p in native.parameters()),
                      'logits_shape': list(output.logits.shape), 'loss': output.loss.item(),
                      'valid_target_tokens': result.metrics['loss'].denominator,
                      'shared_embeddings': native.lm_head.weight is native.model.embed_tokens.weight}))
    if args.export:
        native.save_pretrained(args.export)
        restored = load_qwen3_local(args.export).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(ids).logits, output.logits, rtol=0, atol=0)
        print(f'Local HF export and exact FP32 roundtrip: {args.export}')


if __name__ == '__main__':
    main()
