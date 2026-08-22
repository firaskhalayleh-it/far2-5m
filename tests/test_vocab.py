from far2.data.vocab import (MAX_SIG, VOCAB, digit_significance, num_tokens,
                             parse_num)


def test_size_and_uniqueness():
    assert len(VOCAB) == 72
    assert len(set(VOCAB.names)) == 72


def test_round_trip():
    tokens = ["F2", "BOS", "D4", "D1", "PLUS", "D9", "Q", "ANS", "ANS", "EOS"]
    assert VOCAB.decode(VOCAB.encode(tokens)) == tokens


def test_special_ids_stable():
    assert VOCAB.t("PAD") == 0
    assert VOCAB.t("ANS") == 4


def test_num_tokens_msd_first():
    assert num_tokens(41) == ["D4", "D1"]
    assert num_tokens(0) == ["D0"]
    assert num_tokens(7, width=3) == ["D0", "D0", "D7"]
    assert num_tokens(-305) == ["NEG", "D3", "D0", "D5"]


def test_num_tokens_bases():
    assert num_tokens(10, base=7) == ["D1", "D3"]
    assert num_tokens(10, base=11) == ["DA"]
    for base in (5, 6, 7, 9, 10, 11):
        for v in (0, 1, base - 1, base, base * base + 3, -17):
            assert parse_num(num_tokens(v, base=base), base=base) == v


def test_parse_num_rejects_out_of_base():
    assert parse_num(["DA"], base=10) is None
    assert parse_num(["D7"], base=5) is None
    assert parse_num(["PLUS"]) is None


def test_digit_significance_runs():
    # each contiguous run counts down to 1 at its least-significant digit
    toks = ["D1", "D2", "D3", "PLUS", "D4", "D5", "Q", "ANS"]
    assert digit_significance(toks) == [3, 2, 1, 0, 2, 1, 0, 0]
    long = ["D1"] * (MAX_SIG + 5)
    assert max(digit_significance(long)) == MAX_SIG
