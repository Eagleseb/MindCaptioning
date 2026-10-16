import nbformat as nbf

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(
    """
# Captions from decoded brain activity

The ridge model from `train-decoder` maps each test-perception scan of S1 to DeBERTa-large layer 24 caption features, restored to the raw feature scale. The prefix captioner from `train-captioner` turns a vector in that space into text with a frozen `distilgpt2`.

The captioner was trained on caption features, not on brain-decoded ones. Its inputs were the masked-mean DeBERTa vectors of each video's own reference captions. The 72 test-perception videos were left out of both the ridge fit and the captioner's training and early stopping.

Every condition below generates greedy captions for the 72 test videos and scores them against each video's 20 reference captions:

- **Caption features** feeds the true feature vector of the shown video. This is the reconstruction upper bound.
- **Brain, averaged repetitions** feeds the mean of the five decoded scans of each video.
- **Brain, norm-matched** rescales that mean around the training-feature mean to the average training norm. Ridge shrinks predictions toward the mean, and the captioner never saw vectors that small.
- **Brain, single scans** feeds each of the 360 decoded scans separately.
- **Brain, shuffled videos** pairs each video's references with another test video's decoded vector. It keeps the input distribution but breaks the link to the shown video.
- **Mean training feature** produces one caption for all videos and shows what a generic caption scores.
"""
)

code(
    r'''
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display


def repository_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Open this notebook from the mind_captioning repository")


ROOT = repository_root()
sys.path.insert(0, str(ROOT / "src"))

from mind_captioning.captioner.checkpoint import load_checkpoint
from mind_captioning.captioner.data import VideoCaptionDataset, VideoExample, examples_for_ids, load_examples
from mind_captioning.captioner.generate import generate_captions
from mind_captioning.captioner.loop import make_loader, run_epoch
from mind_captioning.captioner.metrics import rouge_l_max
from mind_captioning.device import pick_device
from train_decoder import average_by_video

SUBJECT = "S1"
FEATURE_MODEL = "deberta-large"
LAYER = 24
DECODER = "distilgpt2"
RUN_DIR = ROOT / "res" / "captioner" / DECODER / FEATURE_MODEL / f"layer{LAYER:02d}"
CHECKPOINT = RUN_DIR / "best"
MAX_NEW_TOKENS = 40
BATCH_SIZE = 16
SEED = 0
'''
)

code(
    r'''
device = pick_device("auto")
model, tokenizer, config, split = load_checkpoint(CHECKPOINT, device)
library = load_examples(
    ROOT / "data" / "caption" / "caption_ck20.csv",
    Path(config["feature_path"]),
    int(config["captions_per_video"]),
)

ridge = np.load(ROOT / "res" / "decoding" / SUBJECT / FEATURE_MODEL / f"layer{LAYER:02d}.npz")
scan_features = ridge["predicted"].astype(np.float32)
scan_ids = ridge["video_id"].astype(np.int64)
test_ids = np.unique(scan_ids)
if set(test_ids.tolist()) != set(split["test"]):
    raise ValueError("The ridge test videos differ from the captioner's test split")
if set(test_ids.tolist()) & (set(split["train"]) | set(split["validation"])):
    raise ValueError("A test-perception video was used to train the captioner")

decoded_mean = average_by_video(scan_features, scan_ids).astype(np.float32)
test_examples = examples_for_ids(library, test_ids)
references = {example.video_id: example.captions for example in test_examples}
true_features = np.stack([example.embedding for example in test_examples])
train_features = np.stack([example.embedding for example in examples_for_ids(library, split["train"])])
train_mean = train_features.mean(axis=0)

print(
    f"Captioner: {config['decoder_name']} prefix model, best epoch {config['best_epoch']}, "
    f"validation loss {config['best_validation_loss']:.4f}"
)
print(
    f"{SUBJECT} test perception: {len(scan_ids)} scans, {len(test_ids)} videos, "
    f"{scan_features.shape[1]} decoded units"
)
print(
    f"Captioner split: {len(split['train'])} train, {len(split['validation'])} validation, "
    f"{len(split['test'])} test videos"
)
'''
)

md(
    """
## Captioner training

Validation loss picked the checkpoint. The test videos played no part in that choice.
"""
)

code(
    r'''
history = json.loads((RUN_DIR / "history.json").read_text())
epochs = [row["epoch"] for row in history]
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
ax.plot(epochs, [row["validation_loss"] for row in history], marker="o", label="validation")
ax.axvline(config["best_epoch"], color="gray", linestyle="--", linewidth=1, label="selected")
ax.set_xlabel("epoch")
ax.set_ylabel("token cross-entropy")
ax.set_title("Prefix captioner on caption features")
ax.legend()
fig.tight_layout()
plt.show()
'''
)

md(
    """
## How close are the decoded vectors to the caption features?

Norms and cosines are measured around the training-feature mean, because every raw vector shares a large common component. A cosine of 1 would mean the decoded vector points exactly at the true caption features of the shown video.
"""
)

code(
    r'''
def centered(values: np.ndarray) -> np.ndarray:
    return values - train_mean


def centered_norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(centered(values), axis=1)


def centered_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = centered(left)
    b = centered(right)
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


reference_norm = float(centered_norm(train_features).mean())
norm_matched = train_mean + centered(decoded_mean) * (reference_norm / centered_norm(decoded_mean))[:, None]
true_by_scan = true_features[np.searchsorted(test_ids, scan_ids)]

rows = [
    ("Training caption features", centered_norm(train_features), None),
    ("Test caption features", centered_norm(true_features), np.ones(len(test_ids))),
    ("Brain, averaged repetitions", centered_norm(decoded_mean), centered_cosine(decoded_mean, true_features)),
    ("Brain, norm-matched", centered_norm(norm_matched), centered_cosine(norm_matched, true_features)),
    ("Brain, single scans", centered_norm(scan_features), centered_cosine(scan_features, true_by_scan)),
]
lines = [
    "| Vectors | Mean centered norm | Mean cosine with the shown video's caption features |",
    "|---|---:|---:|",
]
for name, norms, cosines in rows:
    cosine = "" if cosines is None else f"{cosines.mean():.3f}"
    lines.append(f"| {name} | {norms.mean():.2f} | {cosine} |")
display(Markdown("\n".join(lines)))
'''
)

md(
    """
## Caption scores

- **Loss** is the captioner's teacher-forced cross-entropy on the shown video's 20 references, conditioned on the input vector. Lower means the vector makes the true captions more likely.
- **ROUGE-L, shown** is the best ROUGE-L F1 between the generated caption and any reference of the shown video.
- **ROUGE-L, other** averages the same score over the other 71 test videos. The gap between the two measures how specific a caption is.
- **Identification** ranks the 72 test videos by ROUGE-L between the generated caption and each video's references. A video counts as correct when its own references rank first. Ties share the rank, so a caption that is the same for every video lands at chance.
- **Distinct** counts distinct captions per generated caption. Low values mean generic output.
"""
)

code(
    r'''
column = {int(video_id): index for index, video_id in enumerate(test_ids)}


def caption_all(features: np.ndarray) -> list[str]:
    texts: list[str] = []
    for start in range(0, len(features), BATCH_SIZE):
        texts.extend(
            generate_captions(model, tokenizer, features[start : start + BATCH_SIZE], max_new_tokens=MAX_NEW_TOKENS)
        )
    return texts


def teacher_forced_loss(features: np.ndarray, video_ids: np.ndarray) -> float:
    examples = [
        VideoExample(int(video_id), np.asarray(vector, dtype=np.float32), references[int(video_id)])
        for vector, video_id in zip(features, video_ids)
    ]
    loader = make_loader(VideoCaptionDataset(examples), tokenizer, int(config["max_length"]), 8, False, SEED)
    return run_epoch(model, loader, device)


def rouge_matrix(texts: list[str]) -> np.ndarray:
    cache: dict[str, np.ndarray] = {}
    rows = []
    for text in texts:
        if text not in cache:
            cache[text] = np.array([rouge_l_max(text, references[int(video_id)]) for video_id in test_ids])
        rows.append(cache[text])
    return np.vstack(rows)


def score(name: str, features: np.ndarray, video_ids: np.ndarray, note: str) -> dict:
    texts = caption_all(features)
    scores = rouge_matrix(texts)
    own_columns = np.array([column[int(video_id)] for video_id in video_ids])
    own = scores[np.arange(len(texts)), own_columns]
    others = (scores.sum(axis=1) - own) / (scores.shape[1] - 1)
    higher = (scores > own[:, None]).sum(axis=1)
    ties = (scores == own[:, None]).sum(axis=1) - 1
    ranks = 1 + higher + 0.5 * ties
    loss = teacher_forced_loss(features, video_ids)
    return {
        "name": name,
        "note": note,
        "texts": texts,
        "video_ids": np.asarray(video_ids),
        "own": own,
        "loss": loss,
        "rouge_own": float(own.mean()),
        "rouge_other": float(others.mean()),
        "accuracy": float((ranks == 1).mean()),
        "median_rank": float(np.median(ranks)),
        "distinct": len(set(texts)) / len(texts),
        "n": len(texts),
    }


rng = np.random.default_rng(SEED)
while True:
    permutation = rng.permutation(len(test_ids))
    if not np.any(permutation == np.arange(len(test_ids))):
        break

conditions = [
    score("Caption features", true_features, test_ids, "upper bound"),
    score("Brain, averaged repetitions", decoded_mean, test_ids, "5 scans per video"),
    score("Brain, norm-matched", norm_matched.astype(np.float32), test_ids, "training norm"),
    score("Brain, single scans", scan_features, scan_ids, "360 scans"),
    score("Brain, shuffled videos", decoded_mean[permutation], test_ids, "wrong video"),
    score(
        "Mean training feature",
        np.repeat(train_mean[None, :].astype(np.float32), len(test_ids), axis=0),
        test_ids,
        "one caption",
    ),
]

lines = [
    "| Input | Captions | Loss | Perplexity | ROUGE-L, shown | ROUGE-L, other | Identification | Median rank | Distinct |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in conditions:
    lines.append(
        f"| {row['name']} ({row['note']}) | {row['n']} | {row['loss']:.3f} | {np.exp(row['loss']):.1f} | "
        f"{row['rouge_own']:.3f} | {row['rouge_other']:.3f} | {row['accuracy']:.1%} | "
        f"{row['median_rank']:.1f} of 72 | {row['distinct']:.0%} |"
    )
lines.append("")
lines.append(f"Chance identification is {1 / len(test_ids):.1%}, with a median rank of 36.5.")
display(Markdown("\n".join(lines)))
'''
)

code(
    r'''
by_name = {row["name"]: row for row in conditions}
output = RUN_DIR / "evaluations" / f"brain_{SUBJECT}_testPerception.jsonl"
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w") as handle:
    for index, video_id in enumerate(test_ids):
        record = {"video_id": int(video_id), "reference_captions": list(references[int(video_id)])}
        for name in ("Caption features", "Brain, averaged repetitions", "Brain, norm-matched"):
            row = by_name[name]
            record[name] = {"caption": row["texts"][index], "rouge_l": float(row["own"][index])}
        handle.write(json.dumps(record) + "\n")
print(f"Wrote {output}")

fig, ax = plt.subplots(figsize=(6, 3.5))
for name in ("Caption features", "Brain, averaged repetitions", "Brain, norm-matched", "Brain, shuffled videos"):
    ax.hist(by_name[name]["own"], bins=np.linspace(0, 1, 21), histtype="step", linewidth=1.5, label=name)
ax.set_xlabel("best ROUGE-L F1 against the shown video's references")
ax.set_ylabel("test videos")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
'''
)

md(
    """
## Examples

The videos below are sorted by the ROUGE-L of the averaged brain caption: the two best, two from the middle, and the two worst.
"""
)

code(
    r'''
brain = by_name["Brain, averaged repetitions"]
matched = by_name["Brain, norm-matched"]
upper = by_name["Caption features"]
order = np.argsort(-brain["own"])
middle = len(order) // 2
picked = list(order[:2]) + list(order[middle - 1 : middle + 1]) + list(order[-2:])

blocks = []
for index in picked:
    video_id = int(test_ids[index])
    refs = "\n".join(f"- {text}" for text in references[video_id][:3])
    blocks.append(
        "\n".join(
            [
                f"### Video {video_id}",
                "",
                "**Reference captions**",
                "",
                refs,
                "",
                f"- **Caption features:** {upper['texts'][index]} (ROUGE-L {upper['own'][index]:.2f})",
                f"- **Brain, averaged:** {brain['texts'][index]} (ROUGE-L {brain['own'][index]:.2f})",
                f"- **Brain, norm-matched:** {matched['texts'][index]} (ROUGE-L {matched['own'][index]:.2f})",
            ]
        )
    )
display(Markdown("\n\n".join(blocks)))
'''
)

code(
    r'''
from collections import Counter

lines = ["| Input | Most frequent captions |", "|---|---|"]
for name in ("Brain, averaged repetitions", "Brain, norm-matched", "Mean training feature"):
    common = Counter(by_name[name]["texts"]).most_common(3)
    text = "<br>".join(f"{count}x {caption}" for caption, count in common)
    lines.append(f"| {name} | {text} |")
display(Markdown("\n".join(lines)))
'''
)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "notebooks/brain_captioning.ipynb")
