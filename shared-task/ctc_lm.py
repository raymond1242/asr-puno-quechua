"""
Character n-gram LM and CTC beam search for SIMBig 2026 Task 1.

Why characters, not words. 77% of distinct words in this corpus occur once,
so a word LM has never seen the right form of most words it would need to
score. A character n-gram of order 8-12 spans one or two morphemes: it learns
which suffixes follow which and generalises to unseen words. It is the
subword model IMPROVEMENTS.md item 3 asks for, matched to the character
vocabulary the CTC head already emits.

Why flashlight's lexicon-free decoder, not pyctcdecode. pyctcdecode applies
its LM only when a word is complete, against a word vocabulary. The lexicon-
free decoder scores the LM on every emitted symbol, word delimiter included,
which is what a character LM needs.

Why the ARPA is written here, not by KenLM's `lmplz`. lmplz needs Boost and
CMake to build; this needs neither. `build_arpa` implements the same
interpolated modified Kneser-Ney (Chen & Goodman 1998, as lmplz does it), and
`check_arpa` loads the result in KenLM and verifies that every sampled
context's distribution sums to one.
"""

import json
import math
import random
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

BOS, EOS, UNK, SIL = "<s>", "</s>", "<unk>", "|"

# n-grams are handled as plain strings, one character per symbol, which makes
# counting a substring slice. Sentence markers get a private character each.
_BOS, _EOS = "\x02", "\x03"


def text_to_symbols(text):
    """'simi runa' -> 'simi|runa': the CTC vocabulary's word delimiter."""
    return SIL.join(text.split())


def build_arpa(texts, order, path):
    """Write an interpolated modified Kneser-Ney character LM in ARPA format.

    `texts` are normalised transcripts, words separated by spaces.
    Returns the number of n-grams per order.
    """
    lines = [_BOS + text_to_symbols(t) + _EOS for t in texts if t.strip()]

    # ---- raw counts, every order
    raw = [None] + [Counter() for _ in range(order)]
    for s in lines:
        for n in range(1, order + 1):
            c = raw[n]
            for i in range(len(s) - n + 1):
                g = s[i:i + n]
                if g != _BOS:  # <s> is never predicted
                    c[g] += 1

    # ---- adjusted counts: the highest order keeps raw counts; lower orders
    # use continuation counts (distinct left neighbours), except n-grams that
    # start with <s>, which nothing can precede.
    adj = [None] * (order + 1)
    adj[order] = raw[order]
    for n in range(order - 1, 0, -1):
        cont = Counter()
        for g in raw[n + 1]:
            cont[g[1:]] += 1
        adj[n] = Counter({g: (raw[n][g] if g[0] == _BOS else cont[g])
                          for g in raw[n]})

    # ---- discounts per order, from counts of counts
    discounts = [None]
    for n in range(1, order + 1):
        t = Counter(v for v in adj[n].values() if v <= 4)
        try:
            y = t[1] / (t[1] + 2 * t[2])
            d = (0.0, 1 - 2 * y * t[2] / t[1], 2 - 3 * y * t[3] / t[2],
                 3 - 4 * y * t[4] / t[3])
            if not (0 < d[1] < 1 and 0 < d[2] < 2 and 0 < d[3] < 3):
                raise ValueError
        except (ZeroDivisionError, ValueError):
            d = (0.0, 0.5, 1.0, 1.5)  # too few n-grams to estimate; lmplz's
            # --discount_fallback uses the same values
        discounts.append(d)

    # ---- per-context totals and backoff (gamma) weights
    def disc(n, a):
        return discounts[n][min(a, 3)]

    denom, gamma = {}, {}
    for n in range(1, order + 1):
        tot, mass = defaultdict(int), defaultdict(float)
        for g, a in adj[n].items():
            h = g[:-1]
            tot[h] += a
            mass[h] += disc(n, a)
        for h in tot:
            denom[h] = tot[h]
            gamma[h] = mass[h] / tot[h]

    # Uniform floor for unigrams: every predictable symbol plus <unk>.
    vocab = {g for g in adj[1]} | {UNK}
    uniform = 1.0 / len(vocab)

    prob = {}
    for n in range(1, order + 1):
        for g, a in adj[n].items():
            h = g[:-1]
            lower = prob[g[1:]] if n > 1 else uniform
            prob[g] = (a - disc(n, a)) / denom[h] + gamma[h] * lower
    p_unk = gamma[""] * uniform

    # ---- write
    def tok(g):
        return " ".join(BOS if c == _BOS else EOS if c == _EOS else c for c in g)

    def lg(x):
        return math.log10(x) if x > 0 else -99.0

    counts = [len(adj[n]) + (2 if n == 1 else 0) for n in range(1, order + 1)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        for n, c in enumerate(counts, 1):
            f.write(f"ngram {n}={c}\n")
        for n in range(1, order + 1):
            f.write(f"\n\\{n}-grams:\n")
            if n == 1:
                f.write(f"{lg(p_unk):.6f}\t{UNK}\t0\n")
                f.write(f"-99\t{BOS}\t{lg(gamma[_BOS]):.6f}\n")
            for g in sorted(adj[n]):
                line = f"{lg(prob[g]):.6f}\t{tok(g)}"
                if n < order and g in gamma:
                    line += f"\t{lg(gamma[g]):.6f}"
                f.write(line + "\n")
        f.write("\n\\end\\\n")
    return counts


def check_arpa(path, texts, samples=200, seed=0):
    """Load the ARPA in KenLM; check P(.|context) sums to one.

    Half the contexts are training-text prefixes of any length (sentence-
    initial and mid-sentence states), half random symbol strings (unseen
    contexts, so every backoff path gets exercised). Returns the worst
    deviation from 1 found.
    """
    import kenlm
    m = kenlm.Model(str(path))
    chars = sorted({c for t in texts for c in text_to_symbols(t)})
    syms = chars + [EOS]
    rng = random.Random(seed)
    lines = [text_to_symbols(t) for t in texts if t.strip()]
    worst = 0.0
    for i in range(samples):
        if i % 2:
            s = rng.choice(lines)
            ctx = list(s[:rng.randint(0, len(s))])
        else:
            ctx = rng.choices(chars, k=rng.randint(1, 2 * m.order))
        state, nxt = kenlm.State(), kenlm.State()
        m.BeginSentenceWrite(state)
        for c in ctx:
            m.BaseScore(state, c, nxt)
            state, nxt = nxt, state
        total = sum(10 ** m.BaseScore(state, w, kenlm.State()) for w in syms + [UNK])
        worst = max(worst, abs(total - 1))
    return worst


class CTCBeamDecoder:
    """Beam search over CTC log-probabilities with an optional character LM.

    lm_weight is the LM scale (alpha); sil_score the bonus per word delimiter
    (beta, the usual word insertion bonus). With no LM it reduces to the best
    path, i.e. greedy decoding -- which makes a useful sanity check.
    """

    def __init__(self, vocab, lm_path=None, lm_weight=0.0, sil_score=0.0,
                 beam=64, beam_threshold=25.0, blank="<pad>"):
        from flashlight.lib.text.decoder import (
            CriterionType, KenLM, LexiconFreeDecoder,
            LexiconFreeDecoderOptions, ZeroLM,
        )
        from flashlight.lib.text.dictionary import Dictionary

        self.vocab = list(vocab)
        self.blank = self.vocab.index(blank)
        self.sil = self.vocab.index(SIL)
        self.skip = {i for i, t in enumerate(self.vocab)
                     if t in (blank, BOS, EOS, UNK)}
        # In lexicon-free decoding the LM's "words" are the tokens themselves,
        # so the token dictionary doubles as the LM's.
        lm = KenLM(str(lm_path), Dictionary(self.vocab)) if lm_path else ZeroLM()
        opts = LexiconFreeDecoderOptions(
            beam_size=beam, beam_size_token=len(self.vocab),
            beam_threshold=beam_threshold, lm_weight=lm_weight,
            sil_score=sil_score, log_add=False,
            criterion_type=CriterionType.CTC,
        )
        self._lm = lm  # keep alive: the decoder holds a raw pointer to it
        self.has_lm = bool(lm_path)
        self.decoder = LexiconFreeDecoder(opts, lm, self.sil, self.blank, [])

    def decode(self, logprobs):
        """logprobs: float32 array (frames, vocab), already log-softmaxed."""
        import numpy as np
        e = np.ascontiguousarray(logprobs, dtype=np.float32)
        best = self.decoder.decode(e.ctypes.data, e.shape[0], e.shape[1])[0]
        ids = [i for i, _ in groupby(best.tokens) if i >= 0 and i not in self.skip]
        text = "".join(" " if i == self.sil else self.vocab[i] for i in ids)
        return " ".join(text.split())


def ctc_logprobs(model, processor, audios, device):
    """(frames, vocab) log-probabilities per clip, each trimmed to its length.

    The trim matters: over the zero padding the model still emits symbols,
    which would land at the end of the shorter clips' transcripts.
    """
    import torch
    inputs = processor(audios, sampling_rate=16000, return_tensors="pt",
                       padding=True, return_attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        lp = torch.log_softmax(model(**inputs).logits.float(), dim=-1)
    frames = model._get_feat_extract_output_lengths(
        inputs["attention_mask"].sum(-1)).tolist()
    lp = lp.cpu().numpy()
    return [lp[j, :int(frames[j])] for j in range(len(frames))]


def decoders_from_config(config_path, vocab, lm_dir=None):
    """{domain: CTCBeamDecoder} from the .best.json that 07_tune_lm.py writes.

    `"lm": null` decodes that domain with no LM, which is the greedy best path.
    """
    config_path = Path(config_path)
    lm_dir = Path(lm_dir) if lm_dir else config_path.parent
    cfg = json.loads(config_path.read_text())
    return {
        domain: CTCBeamDecoder(vocab, lm_path=lm_dir / f"{c['lm']}.arpa" if c.get("lm") else None,
                               lm_weight=float(c["alpha"]), sil_score=float(c["beta"]),
                               beam=int(c["beam"]))
        for domain, c in cfg.items()
    }
