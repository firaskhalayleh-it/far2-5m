"""The single shared mathematical vocabulary (v0.4). All families and all
four models use exactly this table — the comparability contract depends on it.

Numbers are DIGIT SEQUENCES (most-significant first), never atomic tokens:
41 -> "D4 D1". A 100x100 lookup surface is impossible; unseen lengths are
representable. digit_significance() computes the Abacus-style significance
index (least-significant digit = 1) that every model adds as a shared
embedding term — no baseline is handicapped by numeric representation."""

SPECIALS = ["PAD", "BOS", "EOS", "SEP", "ANS", "UNDERDETERMINED",
            "NO_SOLUTION", "YES", "NO", "POS"]
MARKERS = ["Q", "CTX", "GOAL", "MIN", "SUBJ", "BASE", "EQUIV", "DERIV",
           "APPLY", "VALID", "FN"]
OPS = ["PLUS", "MINUS", "TIMES", "DIV", "POW", "EQ", "GEQ", "LEQ", "NEQ",
       "SQRT", "MOD", "LPAR", "RPAR", "NEG"]
N_FAMILIES, N_DIGITS, N_VARS = 10, 11, 16      # DA = digit value 10 (base 11)
MAX_SIG = 40                                    # digit-significance embedding rows


class Vocab:
    def __init__(self):
        names: list[str] = []
        names += SPECIALS
        names += MARKERS
        names += OPS
        names += [f"D{i}" for i in range(10)] + ["DA"]
        names += [f"F{i}" for i in range(N_FAMILIES)]
        names += [f"X{i}" for i in range(N_VARS)]
        self.names = names
        self.ids = {n: i for i, n in enumerate(names)}
        assert len(self.ids) == len(names), "duplicate token names"

    def __len__(self) -> int:
        return len(self.names)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.ids[t] for t in tokens]

    def decode(self, ids) -> list[str]:
        return [self.names[int(i)] for i in ids]

    def t(self, name: str) -> int:
        return self.ids[name]


VOCAB = Vocab()
assert len(VOCAB) == 72, f"vocabulary drifted: {len(VOCAB)} tokens"

DIGIT_TOKENS = tuple(f"D{i}" for i in range(10)) + ("DA",)
_DIGIT_VALUE = {t: i for i, t in enumerate(DIGIT_TOKENS)}


def digit_tok(v: int) -> str:
    return DIGIT_TOKENS[v]


def num_tokens(v: int, width: int | None = None, base: int = 10) -> list[str]:
    """v as most-significant-first digit tokens, zero-padded to `width` so
    answer-slot count never leaks magnitude. Negative v gets a NEG prefix
    (NEG is not counted in width)."""
    neg, v = v < 0, abs(v)
    digits = []
    while v:
        digits.append(digit_tok(v % base))
        v //= base
    digits = digits[::-1] or [digit_tok(0)]
    if width is not None:
        assert len(digits) <= width, f"{digits} wider than {width}"
        digits = [digit_tok(0)] * (width - len(digits)) + digits
    return (["NEG"] if neg else []) + digits


def parse_num(tokens: list[str], base: int = 10) -> int | None:
    """Inverse of num_tokens over a digit run (optionally NEG-prefixed)."""
    if not tokens:
        return None
    sign = 1
    if tokens[0] == "NEG":
        sign, tokens = -1, tokens[1:]
    v = 0
    for t in tokens:
        if t not in _DIGIT_VALUE or _DIGIT_VALUE[t] >= base:
            return None
        v = v * base + _DIGIT_VALUE[t]
    return sign * v if tokens else None


def digit_significance(tokens: list[str]) -> list[int]:
    """Abacus-style significance index per position: within each contiguous
    digit run, the least-significant digit gets 1, increasing leftward;
    non-digit positions get 0. Shared by ALL models (see models/layers)."""
    sig = [0] * len(tokens)
    i = 0
    while i < len(tokens):
        if tokens[i] in _DIGIT_VALUE:
            j = i
            while j < len(tokens) and tokens[j] in _DIGIT_VALUE:
                j += 1
            for k in range(i, j):
                sig[k] = min(j - k, MAX_SIG)
            i = j
        else:
            i += 1
    return sig
