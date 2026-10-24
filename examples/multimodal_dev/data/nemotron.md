# Nemotron Image v3 for MDP

`--dataset-provider nemotron` supplies complete image conversations to the
Qwen3.5-VL MDP example. SEC and CLEVR use the same reader and admission checks.
The provider preserves conversation/image order, creates assistant-only
next-token targets, and returns the native MDP sample dictionary. Native
samplers, greedy packing, static THD collation, and model scheduling are unchanged.

## Prepared input contract

Download, conversion, length assessment, and experiment/report tooling live
outside Megatron-LM, in `agentic-mcore-dev/scripts/nemotron/` (download entry:
`scripts/nemotron_download.py`). Training does not import that repository,
Megatron-Bridge, or Energon. Any trusted producer implementing the following
contract can supply the prepared artifacts; no Energon loader or `.nv-meta`
index is consumed.

A prepared directory contains:

```text
prepared/
  train-shard-000000.tar
  val-shard-000000.tar
  manifest.json
  mdp-training-index.json
  mdp-selection-clevr_2-32768.json
```

Each record has a `<subset>__<id>.json` member containing ChatML messages and a
matching `.jpgs` member containing a trusted pickle list of original image bytes
in message order. Only use artifacts from a trusted preparation pipeline.

| Artifact | Required contract |
|---|---|
| `manifest.json` | Verified source provenance, dataset `revision`, `source_integrity_verified=true`, subset names and split counts |
| `mdp-training-index.json` | `format_version=1`, identical `manifest`, and `splits.train`/`splits.val` record arrays |
| Index record | `key`, relative `tar` filename, and `[byte_offset, byte_count]` extents for `json` and `jpgs` |
| Selection JSON | `kind=nemotron-mdp-complete-record-selection`, `format_version=1`, identical `manifest`, `subset`, `token_budget`, processor settings, split key lists, and `counts.<split>.selected_records` |

The supported dataset revision is `7656391d4d4cb11ec3722b34f10d499435de0460`.
Processor settings are `processor=Qwen/Qwen3.5-35B-A3B`,
`processor_revision=59d61f3ce65a6d9863b86d2e96597125219dc754`,
`min_pixels=4096`, and `max_pixels=262144`. Cache that processor before launching;
workers use `local_files_only=True`. The example uses vocabulary 248320 and
image token ID 248056.

The producer must verify source files, count every complete record with the same
processor settings, and admit only records within the requested token budget.
Apply this procedure to both SEC and CLEVR, even when all records fit. Exclude
an overlong record whole; do not truncate conversations or discard images.
Validate representative actual Dataset outputs, including near-budget records,
before publishing the selection.

Training verifies matching provenance, processor settings, exact budget, selected
counts, and unique keys belonging to the requested subset and split. Every
loaded record is checked for token budget, vocabulary bounds, image/grid agreement,
and nonempty assistant targets. It fails with the source key on invalid input.
It does not rehash source tars or rerun the full census in each worker; keep
prepared artifacts immutable. Receipt files from preparation are audit outputs,
not additional runtime dependencies.

## Use with a configured Qwen3.5-VL MDP launch

Add the data and packing arguments below to the standard
`examples.multimodal_dev.pretrain_multimodal` launch with your model, optimizer,
and parallelism configuration. Choose either the CLEVR or SEC selection path.

```bash
--dataset-provider nemotron \
--data-path /data/prepared/mdp-selection-clevr_2-32768.json \
--seq-length 32768 \
--use-packed-sequence \
--use-vanilla-collate-fn \
--dataloader-type cyclic \
--mdp-enable \
--mdp-greedy-packing \
--thd-static-packing \
--pad-packed-seq-alignment max \
--max-seqlen-per-dp-cp-rank 32768 \
--thd-max-packed-sequences 64
```

For SEC 2, use its `mdp-selection-long_document_sec_2-32768.json` instead.
Data selection does not choose model size, topology, or an experiment schedule.
Native samplers partition/shuffle virtual indices; repetition remains within
each split. The virtual length provisions enough records for greedy's maximum
records per pack and lookahead. There is no separate test split. Greedy exact
resume and weighted subset mixing are not added by this provider.

## Validation

```bash
python -m pytest -q examples/multimodal_dev/tests/test_nemotron_dataset.py
```

The CPU tests use independently constructed prepared artifacts and a deterministic
processor stub; they require the normal Megatron test dependencies, including
PyTorch and Pillow. They cover ordered reads, supervision, budget/provenance
rejection and split isolation. They do not certify model forward/backward or
performance; validate a new training environment with a GPU smoke run.


For buffered FFD, replace `--mdp-greedy-packing` with `--mdp-ffd-packing` and
optionally set `--mdp-ffd-packing-buffer-size` (default 128 samples). Native
sampler ownership and complete-record admission stay unchanged. Both policies
provision the same virtual dataset length, preserving source order for a
controlled comparison; FFD changes bin assignment within each buffer. Both
are benchmark paths without exact checkpoint resume.
