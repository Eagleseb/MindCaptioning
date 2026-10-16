# Result summaries

This directory is the runtime destination for decoder weights, checkpoints,
logs, generated captions, and evaluations. Those artifacts can be large and
are ignored by Git.

The publication keeps only:

- `captioner/summary.json`: aggregate metrics across the retained experiments;
- `shells.json`: local feature-space robustness metrics;
- `cell_diagnostic.json`: decoded-vector cell membership statistics;
- `captioner/runs/index.jsonl`: one compact ledger entry per experiment.

The summaries do not contain the data or model weights needed to reproduce a
run. See the root README for the expected local data layout and commands.