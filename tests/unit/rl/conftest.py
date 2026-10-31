import json
from pathlib import Path
import pytest
import torch
from mahjong.app.config import parse_model
from mahjong.models.qwen3 import build_qwen3, Qwen3PolicyAdapter, SETTINGS_FIELDS
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA

@pytest.fixture
def tiny():
    torch.set_num_threads(1)
    config = parse_model(['model.num_hidden_layers=1','model.hidden_size=8',
        'model.intermediate_size=16','model.num_attention_heads=1','model.num_key_value_heads=1',
        'model.head_dim=8','model.vocab_size=1024','model.max_position_embeddings=8192'])
    settings = {k:config['model'][k] for k in SETTINGS_FIELDS}
    settings['input_schema'] = CONTEXT_SCHEMA
    actor = Qwen3PolicyAdapter(build_qwen3(settings,init_seed=42),input_schema=CONTEXT_SCHEMA)
    profile = json.loads(Path('configs/inference-profile.json').read_text())
    return actor, settings, profile
