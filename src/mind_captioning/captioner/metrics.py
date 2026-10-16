"""Caption scores for reconstruction from caption-derived embeddings.

ROUGE-L measures lexical overlap with the references. It misses paraphrases
that share few words. BERTScore, when the optional package is installed,
compares contextual token similarities and still scores text against text.
Neither score measures visual grounding or brain decoding.
"""

from __future__ import annotations


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 between one hypothesis and one reference."""
    hyp = _tokens(hypothesis)
    ref = _tokens(reference)
    if not hyp or not ref:
        return 0.0
    match = _lcs_length(hyp, ref)
    precision = match / len(hyp)
    recall = match / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l_max(hypothesis: str, references: list[str] | tuple[str, ...]) -> float:
    """Best ROUGE-L F1 against any reference caption of the same video."""
    if not references:
        raise ValueError("rouge_l_max needs at least one reference")
    return max(rouge_l(hypothesis, reference) for reference in references)


def maybe_bertscore_f1(
    hypotheses: list[str],
    references: list[list[str]],
) -> list[float] | None:
    """Return per-hypothesis BERTScore F1, or None when the package is absent.

    Multiple references contribute through the maximum F1 for that hypothesis.
    """
    try:
        import bert_score
    except ImportError:
        return None
    scores: list[float] = []
    for hypothesis, group in zip(hypotheses, references, strict=True):
        expanded = [hypothesis] * len(group)
        _, _, f1 = bert_score.score(expanded, list(group), lang="en", verbose=False)
        scores.append(float(f1.max()))
    return scores
