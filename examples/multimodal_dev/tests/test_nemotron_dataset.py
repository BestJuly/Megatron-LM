# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regressions for the Nemotron records consumed by native MDP packing."""
import io
import json
import pickle
import tarfile
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from examples.multimodal_dev.data import nemotron as provider
from examples.multimodal_dev.data.nemotron import tokenize


class Processor:
    """Expose fixed ChatML tokens with known target spans and one image grid."""

    image_token = '<image>'
    image_processor = SimpleNamespace(merge_size=2)
    # Two assistant turns, a masked user image token, and a special reasoning
    # delimiter inside the first answer. Numbers 20/21/22 are answer targets.
    ids = [1, 11, 9, 30, 2, 1, 10, 3, 20, 21, 2, 1, 11, 31, 2, 1, 10, 22, 2]

    def __init__(self):
        self.tokenizer = self
        self.all_special_ids = [1, 2, 3, 9]

    def apply_chat_template(self, messages, **kwargs):
        """Return rendered text independently of raw answer strings."""
        return 'rendered conversation'

    def convert_tokens_to_ids(self, value):
        """Resolve the role boundaries and image placeholder used by the fixture."""
        return {'<|im_start|>': 1, '<|im_end|>': 2, '<image>': 9}[value]

    def encode(self, text, **kwargs):
        """Supply the tokenized assistant role header."""
        assert text == 'assistant\n'
        return [10]

    def __call__(self, **kwargs):
        return {
            'input_ids': torch.tensor([self.ids]),
            'pixel_values': torch.zeros(4, 12),
            'image_grid_thw': torch.tensor([[1, 2, 2]]),
        }


MESSAGES = [
    {'role': role, 'content': [{'type': 'text', 'text': 'example'}]}
    for role in ('user', 'assistant', 'user', 'assistant')
]


def write_data(root, train_key='source__a', val_key='source__b', colors=((12, 34, 56),)):
    """Write a producer-independent fixture with known paired tar extents."""
    images = []
    for color in colors:
        buf = io.BytesIO()
        Image.new('RGB', (2, 2), color).save(buf, format='PNG')
        images.append(buf.getvalue())
    splits = {}
    for split, key in [('train', train_key), ('val', val_key)]:
        path = root / f'{split}-shard-000000.tar'
        with tarfile.open(path, 'w') as archive:
            for suffix, data in [
                ('jpgs', pickle.dumps(images)),
                ('json', json.dumps(MESSAGES).encode()),
            ]:
                member = tarfile.TarInfo(f'{key}.{suffix}')
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        with tarfile.open(path) as archive:
            extents = {
                suffix: [
                    archive.getmember(f'{key}.{suffix}').offset_data,
                    archive.getmember(f'{key}.{suffix}').size,
                ]
                for suffix in ('json', 'jpgs')
            }
        splits[split] = [dict(key=key, tar=path.name, **extents)]
    manifest = {
        'revision': provider.REVISION,
        'source_integrity_verified': True,
        'counts': {'train': 1, 'val': 1},
        'subsets': [train_key.partition('__')[0]],
    }
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / provider.INDEX).write_text(
        json.dumps({'format_version': 1, 'manifest': manifest, 'splits': splits})
    )
    return manifest


def read_fixture_index(root):
    """Read fixture metadata without importing any offline preparation code."""
    return json.loads((root / provider.INDEX).read_text())


def selection_for(index, subset, splits=None):
    """Describe the public admission format independently of its producer."""
    splits = splits or {name: [r['key'] for r in rows] for name, rows in index['splits'].items()}
    return {
        'kind': 'nemotron-mdp-complete-record-selection',
        'format_version': 1,
        'manifest': index['manifest'],
        'subset': subset,
        'token_budget': 32,
        'processor': provider.PROCESSOR,
        'processor_revision': provider.PROCESSOR_REVISION,
        'min_pixels': 4096,
        'max_pixels': 262144,
        'splits': splits,
        'counts': {name: {'selected_records': len(keys)} for name, keys in splits.items()},
    }


def dataset(root, records, max_length=32):
    """Bind the deterministic processor to a real offset-reading Dataset."""
    ds = provider.NemotronDataset(
        root, records, target_length=10, max_length=max_length, vocab_size=64, image_token_id=9
    )
    ds._processor = Processor()
    return ds


def test_assistant_targets_are_shifted_and_special_tokens_are_excluded():
    """Assert the exact predicted token positions across two assistant turns."""
    sample = tokenize(Processor(), MESSAGES, [Image.new('RGB', (2, 2))])
    expected = [-100] * len(Processor.ids)
    expected[7], expected[8], expected[16] = 20, 21, 22
    assert sample['labels'].tolist() == expected
    assert sample['loss_mask'].nonzero().flatten().tolist() == [7, 8, 16]
    assert sample['input_ids'].tolist() == Processor.ids
    assert sample['pixel_values'].dtype == torch.bfloat16


def test_unclosed_assistant_turn_fails_instead_of_training_unbounded_text():
    """Reject a missing ChatML end delimiter before assigning targets."""
    p = Processor()
    p.ids = p.ids[:-1]
    with pytest.raises(ValueError, match='no closing'):
        tokenize(p, MESSAGES, [])


def test_offset_reads_preserve_pairing_and_repeat_within_split(tmp_path):
    """Resolve reversed tar components and repeat without crossing split boundaries."""
    write_data(tmp_path)
    index = read_fixture_index(tmp_path)
    ds = dataset(tmp_path, index['splits']['train'])
    assert len(ds) == 10
    assert torch.equal(ds[0]['labels'], ds[9]['labels'])
    assert ds.__getstate__()['_files'] == {}
    assert ds.__getstate__()['_processor'] is None


@pytest.mark.parametrize('problem', ['over_budget', 'vocabulary', 'image_grid'])
def test_bad_training_record_fails_with_key_instead_of_silent_replacement(tmp_path, problem):
    """Report invalid token budgets, vocabulary and image grids with the source key."""
    write_data(tmp_path)
    ds = dataset(
        tmp_path,
        read_fixture_index(tmp_path)['splits']['train'],
        max_length=16 if problem == 'over_budget' else 32,
    )
    if problem == 'vocabulary':
        ds.vocab_size = 30
    if problem == 'image_grid':
        ds.image_token_id = 8
    with pytest.raises(RuntimeError, match='Nemotron record source__a:'):
        ds[0]


def selection_fixture():
    """Provide an external selection containing a subset of indexed records."""
    index = {
        'manifest': {'revision': provider.REVISION},
        'splits': {
            'train': [{'key': 'sec__a'}, {'key': 'sec__long'}, {'key': 'other__x'}],
            'val': [{'key': 'sec__b'}],
        },
    }
    return index, selection_for(index, 'sec', {'train': ['sec__a'], 'val': ['sec__b']})


def test_selection_resolves_keys_without_changing_the_index():
    """Return admitted keys in selection order and leave other records untouched."""
    index, selection = selection_fixture()
    chosen = provider.select_records(index, selection, 32)
    assert chosen == {'train': [{'key': 'sec__a'}], 'val': [{'key': 'sec__b'}]}
    assert index['splits']['train'][1]['key'] == 'sec__long'


@pytest.mark.parametrize(
    'problem', ['budget', 'provenance', 'processor', 'duplicate', 'cross_split', 'empty']
)
def test_selection_rejects_stale_or_invalid_training_views(problem):
    """Reject selection metadata and keys that disagree with training inputs."""
    index, selection = selection_fixture()
    if problem == 'budget':
        selection['token_budget'] = 64
    if problem == 'provenance':
        selection['manifest'] = {}
    if problem == 'processor':
        selection['max_pixels'] = 65536
    if problem == 'duplicate':
        selection['splits']['train'] *= 2
    if problem == 'cross_split':
        selection['splits']['train'] = ['sec__b']
    if problem == 'empty':
        selection['splits']['val'] = []
    with pytest.raises(ValueError):
        provider.select_records(index, selection, 32)


@pytest.mark.parametrize('subset', ['clevr_2', 'long_document_sec_2'])
def test_all_fitting_subsets_still_require_an_explicit_selection(tmp_path, subset):
    """Apply the same required admission contract to single- and multi-image families."""
    write_data(tmp_path, train_key=f'{subset}__a', val_key=f'{subset}__b')
    index = read_fixture_index(tmp_path)
    with pytest.raises(ValueError, match='selection JSON, not a tar directory'):
        provider.load_training_selection(tmp_path, 32)
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(selection_for(index, subset)))
    root, records = provider.load_training_selection(path, 32)
    assert root == tmp_path and records == index['splits']
    with pytest.raises(ValueError, match='budget differs'):
        provider.load_training_selection(path, 64)


@pytest.mark.parametrize('problem', ['revision', 'unverified', 'index_manifest'])
def test_runtime_rejects_unverified_or_changed_provenance(tmp_path, problem):
    """Fail before loading records if producer metadata disagree."""
    manifest = write_data(tmp_path)
    index = read_fixture_index(tmp_path)
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(selection_for(index, 'source')))
    if problem == 'revision':
        manifest['revision'] = 'wrong-revision'
    elif problem == 'unverified':
        manifest['source_integrity_verified'] = False
    else:
        manifest['subsets'] = ['different-source']
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='integrity-verified|provenance differs'):
        provider.load_training_selection(path, 32)


def test_truncated_tar_payload_fails_with_source_key(tmp_path):
    """A broken extent must fail instead of returning a replacement sample."""
    write_data(tmp_path)
    record = read_fixture_index(tmp_path)['splits']['train'][0]
    record['json'] = [0, 1_000_000]
    with pytest.raises(RuntimeError, match='source__a: Truncated prepared component'):
        dataset(tmp_path, [record])[0]


def test_multiple_images_reach_the_processor_in_source_order(tmp_path):
    """Keep both distinct images in their original order through offset reads."""
    colors = ((12, 34, 56), (78, 90, 123))
    write_data(tmp_path, colors=colors)
    ds = dataset(tmp_path, read_fixture_index(tmp_path)['splits']['train'])

    class OrderedProcessor(Processor):
        """Require the ordered images and produce matching visual placeholders."""

        ids = Processor.ids[:3] + [9] + Processor.ids[3:]

        def __call__(self, **kwargs):
            assert tuple(image.getpixel((0, 0)) for image in kwargs['images']) == colors
            encoded = super().__call__(**kwargs)
            encoded['image_grid_thw'] = torch.tensor([[1, 2, 2], [1, 2, 2]])
            encoded['pixel_values'] = torch.zeros(8, 12)
            return encoded

    ds._processor = OrderedProcessor()
    assert ds[0]['image_grid_thw'].shape[0] == 2
