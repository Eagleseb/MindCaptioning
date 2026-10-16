"""Test-split loss, caption scores, and a qualitative JSONL report."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from mind_captioning.captioner import LIMITATION
from mind_captioning.captioner.data import VideoCaptionDataset, examples_for_ids
from mind_captioning.captioner.generate import generate_caption, generate_captions
from mind_captioning.captioner.loop import make_loader, run_epoch
from mind_captioning.captioner.metrics import maybe_bertscore_f1, rouge_l_max


def evaluate_examples(
    model,
    tokenizer,
    examples,
    train_examples,
    output_jsonl: Path,
    device: torch.device,
    max_length: int,
    batch_size: int,
    seed: int,
    max_new_tokens: int,
    num_beams: int = 1,
) -> dict:
    """Score greedy captions against each video's own references.

    The baseline caption is generated once from the mean training embedding.
    Greedy decoding makes that caption identical for every video, so a higher
    conditional score is evidence that the projector uses the input vector.
    """
    if not examples or not train_examples:
        raise ValueError("Evaluation needs evaluation videos and training videos")
    dataset = VideoCaptionDataset(examples, mode="all_captions", seed=seed)
    loader = make_loader(dataset, tokenizer, max_length, batch_size, False, seed)
    loss = run_epoch(model, loader, device)
    embeddings = np.stack([example.embedding for example in examples])
    captions: list[str] = []
    for start in range(0, len(examples), batch_size):
        captions.extend(
            generate_captions(
                model,
                tokenizer,
                embeddings[start : start + batch_size],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )
        )
    mean_embedding = np.stack([example.embedding for example in train_examples]).mean(axis=0)
    baseline = generate_caption(
        model,
        tokenizer,
        mean_embedding.astype(np.float32),
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
    )
    references = [list(example.captions) for example in examples]
    bert = maybe_bertscore_f1(captions, references)
    if bert is None:
        print("BERTScore is unavailable because the bert_score package is not installed.")
    rows = []
    rouge_scores = []
    baseline_scores = []
    for index, example in enumerate(examples):
        rouge = rouge_l_max(captions[index], example.captions)
        baseline_rouge = rouge_l_max(str(baseline), example.captions)
        rouge_scores.append(rouge)
        baseline_scores.append(baseline_rouge)
        row = {
            "video_id": example.video_id,
            "generated_caption": captions[index],
            "reference_captions": list(example.captions),
            "rouge_l": rouge,
            "baseline_caption": baseline,
            "baseline_rouge_l": baseline_rouge,
            "bert_score_f1": None if bert is None else bert[index],
        }
        rows.append(row)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "loss": loss,
        "perplexity": math.exp(loss),
        "rouge_l": float(np.mean(rouge_scores)),
        "baseline_rouge_l": float(np.mean(baseline_scores)),
        "bert_score_f1": None if bert is None else float(np.mean(bert)),
        "n_videos": len(examples),
        "baseline": "greedy caption from the mean training-split embedding",
        "limitation": LIMITATION,
        "metric_notes": {
            "rouge_l": (
                "Maximum ROUGE-L F1 against the video's reference captions. "
                "It counts word overlap and misses paraphrases."
            ),
            "bert_score_f1": (
                "Maximum BERTScore F1 against the reference captions when bert_score is installed. "
                "It compares the generated text with the references and does not see the video."
            ),
            "perplexity": (
                "exp(loss), where loss averages videos, then captions inside a video, then tokens "
                "inside a caption."
            ),
        },
        "jsonl": str(output_jsonl),
    }
    summary_path = output_jsonl.with_name(output_jsonl.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["summary_path"] = str(summary_path)
    return summary


def examples_from_split(library, split: dict, name: str):
    return examples_for_ids(library, split[name])
