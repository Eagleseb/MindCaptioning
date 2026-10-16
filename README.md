# Mind Captioning with a learned feature-to-text decoder

This repository explores a partial reproduction and modification of
[Mind captioning: Evolving descriptive text of mental content from human brain
activity](https://doi.org/10.1126/sciadv.adw1464) by Tomoyasu Horikawa.

The paper uses two stages:

1. linear ridge regression maps whole-brain fMRI activity to semantic caption
   features extracted by a frozen language model;
2. an iterative masked-language-model search replaces and inserts words until a
   candidate sentence aligns with the brain-decoded semantic features.

This project reproduces the first-stage data alignment and ridge decoder, but
deliberately replaces the paper's second stage. Instead of iterative MLM search,
it trains a causal language model decoder to map a semantic feature vector to a
caption:

```text
fMRI -> ridge regression -> DeBERTa caption-feature space
     -> learned prefix projector -> frozen or lightly adapted causal LM -> text
```

This is an experimental alternative, not a faithful reproduction of the
paper's feature-to-text algorithm.

## What is implemented

- Caption feature extraction with a Hugging Face encoder. Token states are
  pooled with a masked mean that excludes padding and special tokens; the 20
  captions for each video are then averaged.
- Subject-specific ridge regression from fMRI activity to caption features.
- A prefix-conditioned causal decoder, using DistilGPT-2 by default.
- Frozen-decoder, LoRA, normalization/noise, and cell-preserving augmentation
  experiments.
- Brain-feature evaluation, shuffled-video controls, ROUGE-L, and video
  identification.

The S1 perception split is defined by the stimulus files: 1,897 videos for
training, 211 for validation, and 72 held-out test videos. Test videos are not
used for fitting or early stopping.

## Important limitation

The captioner is trained primarily on **caption-derived embeddings**. Training
loss and clean caption-feature scores therefore measure reconstruction from
reference-caption features, not direct brain-to-text generation. At inference,
the input comes from a ridge model and has a different distribution.

This mismatch is the main limitation observed in the experiments. Increasing
captioner capacity improved reconstruction of clean caption embeddings but did
not improve—and sometimes reduced—accuracy on brain-decoded vectors.

## Results

The retained summaries are under `res/`; checkpoints, generated captions, logs,
raw evaluations, and datasets are intentionally excluded from Git.

For S1 test perception (72 videos):

| Experiment | Brain input | ROUGE-L | Identification |
| --- | --- | ---: | ---: |
| Frozen baseline | norm-matched | 0.370 | 11.1% |
| Baseline + repetition controls | norm-matched | 0.377 | 12.5% |
| Z-score + shrink/noise | decoded, z-scored | 0.354 | 6.9% |
| 16 prefixes + LoRA | decoded, z-scored | 0.336 | 9.7% |
| Cell-preserving augmentation | norm-matched | **0.388** | 11.1% |

The cell-preserving run improved local robustness and ROUGE-L, but not video
identification. Only 47.2% of norm-matched brain-decoded test vectors fell in
the Voronoi cell of their true caption embedding (19.4% without norm matching).
The generated descriptions remain substantially below the intended quality of
the original Mind Captioning experiment.

## Conclusion and next steps

The current results are not good enough for accurate brain captioning. They
indicate that the feature-to-text model should be trained on inputs that
reproduce the errors and geometry of **brain-decoded features**, rather than
only on clean caption embeddings. A practical next experiment is to generate
out-of-fold ridge predictions for the training stimuli and train the captioner
on those predictions, while keeping the final test set untouched.

A more direct alternative is to remove the intermediate DeBERTa projection and
train a neural fMRI-to-text model end to end. Such a model could learn which
parts of the brain signal are useful for generation instead of forcing all
information through a caption-feature regression bottleneck. It would require
careful regularization, subject handling, and leakage-free validation because
the available fMRI dataset is small relative to modern language models.

## Setup

Python 3.10 or newer is required.

```bash
uv sync
```

The data are not distributed in this repository. Expected inputs include:

```text
data/
├── caption/caption_ck20.csv
├── fmri/data/preprocessed/
│   ├── trainPerception_S1.mat
│   └── testPerception_S1.mat
└── feature/video/deberta-large/layer24.mat
```

## Reproducing the pipeline

Extract caption features:

```bash
uv run extract-video-features \
  --model microsoft/deberta-large \
  --subject S1
```

Fit the fMRI-to-feature ridge decoder:

```bash
uv run train-decoder \
  --model deberta-large \
  --layer 24 \
  --subject S1
```

Train the frozen baseline captioner:

```bash
uv run train-captioner \
  --feature-model deberta-large \
  --layer 24 \
  --decoder distilgpt2
```

Train the cell-preserving experiment:

```bash
uv run python -m run_cell_experiment
```

Run the test suite:

```bash
uv run python -m unittest discover -s tests -q
```

## Repository layout

```text
src/        feature extraction, ridge decoding, captioner training/evaluation
tests/      unit and integration tests
notebooks/  analysis builders; executed notebooks are ignored
res/        compact metric summaries only
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Citation

Horikawa, T. “Mind captioning: Evolving descriptive text of mental content from
human brain activity.” *Science Advances* (2025).
[doi:10.1126/sciadv.adw1464](https://doi.org/10.1126/sciadv.adw1464)
