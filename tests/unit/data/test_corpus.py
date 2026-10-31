"""Unified reading, split isolation, tamper checks and source retirement."""
from collections import Counter
from compression import zstd
from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil

import pytest
import torch

from mahjong.data.mahjong.corpus import PolicyCorpusReader
from mahjong.data.mahjong.corpus_pack import pack_corpus
from mahjong.data.mahjong.cleaning.catalog import codec_binding
from mahjong.data.mahjong.cleaning.common import atomic_json, canonical, digest, file_identity, read_json, CleaningError
from mahjong.data.mahjong.cleaning.encoding import encode_view
from mahjong.data.mahjong.cleaning.text_codec import encode_round
from mahjong.data.mahjong.policy import MahjongPolicyTrainSource, MahjongPolicyEvalSource
from mahjong.policy.context import read_context_prefix
from mahjong.tokenizer import Action, Atom, MahjongTokenizer
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA, PolicyContextCodec


from tests.support.policy import manual_build


@pytest.fixture
def corpus(tmp_path):
    inputs=[manual_build(tmp_path/'old-a','fixture-a'),manual_build(tmp_path/'old-b','fixture-b')]
    output=tmp_path/'training';pack_corpus(inputs,output,workers=1)
    for path in inputs:shutil.rmtree(path)
    return output


def test_unified_reader_needs_only_training_files_and_bounds_open_indices(corpus):
    assert {p.name for p in corpus.iterdir()}=={'dataset.json','games.jsonl.zst','shards'}
    assert not list(corpus.rglob('*.sqlite*')) and not any(p.is_symlink() for p in corpus.rglob('*'))
    with PolicyCorpusReader(corpus,open_shards=1) as reader:
        games=list(reader.iter_games());assert len(games)==4
        for game in games:
            key=game['source_key'];decisions=reader.read_policy_decisions(key,4);assert len(decisions)==1
            ids,_=read_context_prefix(reader,key,4,0,8)
            seq=PolicyContextCodec().decode_policy_prefix(ids)
            assert len(seq.previous_rounds)==3
            assert len(reader._indices)<=1 and len(reader._files)<=1
            assert reader.sources[key]['revision']=='artificial'


def test_mixed_game_split_and_sampler_restore_survive_removal_of_cleaning_trees(corpus):
    kwargs=dict(build_dir=corpus,seed=7,batch_size=2,input_schema=CONTEXT_SCHEMA,
                max_sequence_length=8192,validation_fraction=.5)
    source=MahjongPolicyTrainSource(**kwargs)
    validation=MahjongPolicyEvalSource(**{k:v for k,v in kwargs.items() if k!='seed'})
    try:
        train={source.dataset.games[int(i)][0] for i in source.indices}
        heldout={validation.dataset.games[int(i)][0] for i in validation.indices}
        assert train and heldout and not train.intersection(heldout) and len(train|heldout)==4
        lease=source.acquire();source.commit(lease.token);state=source.state_dict()
        restored=MahjongPolicyTrainSource(**kwargs)
        try:
            restored.load_state_dict(state)
            expected,actual=source.acquire(),restored.acquire()
            assert expected.batch.sample_ids==actual.batch.sample_ids
            for key,value in expected.batch.payload.items():assert torch.equal(value,actual.batch.payload[key])
            source.commit(expected.token);restored.commit(actual.token)
        finally:restored.close()
        assert sum(batch.sample_count for batch in validation.iter_batches())>0
    finally:source.close();validation.close()


@pytest.mark.parametrize('name',['index.json.zst','data.blocks.zst'])
def test_corrupt_published_data_is_rejected(corpus,name):
    path=corpus/'shards/part-000000'/name
    with path.open('r+b') as stream:stream.seek(4);stream.write(b'bad')
    with PolicyCorpusReader(corpus) as reader:
        source=next(reader.iter_games())['source_key']
        with pytest.raises(CleaningError,match='STORAGE_CORRUPT'):reader.read_round(source,0)


def test_overlapping_inputs_are_not_published_or_removed(tmp_path):
    build=manual_build(tmp_path/'old','fixture');output=tmp_path/'training'
    with pytest.raises(CleaningError,match='DUPLICATE_SOURCE'):
        pack_corpus([build,build],output,workers=1)
    assert not output.exists() and (build/'shards/part-000000/data.blocks.zst').is_file()
