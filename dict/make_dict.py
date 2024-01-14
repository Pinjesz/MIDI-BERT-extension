import pickle
import miditok


def make_cp_new(config) -> tuple[miditok.MIDITokenizer, tuple]:
    tokenizer = miditok.CPWord(config)
    vocab = tokenizer.vocab

    event2word = {}
    word2event = {}

    for kind in ["Bar", "Position", "Pitch", "Duration"]:
        event2word[kind] = {
            "Pad_None": 0,
            "Mask_None": 1,
        }
        word2event[kind] = {
            0: "Pad_None",
            1: "Mask_None",
        }

    # Bar
    event2word["Bar"]["Bar_New"] = 2
    event2word["Bar"]["Bar_Continue"] = 3
    word2event["Bar"][2] = "Bar_New"
    word2event["Bar"][3] = "Bar_Continue"

    # Position
    for key, val in list(vocab[1].items())[4:]:
        event2word["Position"][key] = val - 2
        word2event["Position"][val - 2] = key

    # Pitch
    for key, val in list(vocab[2].items())[3:]:
        event2word["Pitch"][key] = val - 1
        word2event["Pitch"][val - 1] = key

    # Duration
    for key, val in list(vocab[4].items())[3:]:
        event2word["Duration"][key] = val - 1
        word2event["Duration"][val - 1] = key

    t = (event2word, word2event)

    return tokenizer, t


def add_to_cp(
    tokenizer: miditok.CPWord, e2w: dict[str, int], token: str
) -> miditok.MIDITokenizer:
    # TODO implement
    return tokenizer, e2w


def make_simple_new(config, tokenizer_name) -> tuple[miditok.MIDITokenizer, tuple]:
    if tokenizer_name == "remi":
        tokenizer = miditok.REMI(config)
    elif tokenizer_name == "tsd":
        tokenizer = miditok.TSD(config)
    elif tokenizer_name == "structured":
        tokenizer = miditok.Structured(config)
    else:
        raise NotImplementedError("This tokenizer type is not implemented")

    vocab = tokenizer.vocab

    event2word = {}
    word2event = {}
    used = 0
    for key, value in vocab.items():
        if key == "PAD_None":
            _key = "Pad_None"
        elif key == "MASK_None":
            _key = "Mask_None"
        else:
            _key = key

        if len(_key) >= 8 and (_key[:8] == "Velocity" or _key[:8] == "Program_"):
            continue

        event2word[_key] = used
        word2event[used] = _key
        used += 1

    with open(f"dict/additional_tokens/{tokenizer_name}_new.txt", "r") as file:
        events = file.readline().split(",")
        for e in events:
            if e == "":
                continue
            tokenizer.add_to_vocab(e)
            event2word[e] = used
            word2event[used] = e
            used += 1

    t = (event2word, word2event)
    return tokenizer, t


def add_to_simple(
    tokenizer: miditok.MIDITokenizer, e2w: dict[str, int], token: str
) -> miditok.MIDITokenizer:
    tokenizer_name = ""
    if type(tokenizer) is miditok.REMI:
        tokenizer_name = "remi"
    elif type(tokenizer) is miditok.TSD:
        tokenizer_name = "tsd"
    elif type(tokenizer) is miditok.Structured:
        tokenizer_name = "structured"
    else:
        raise NotImplementedError("This tokenizer type is not implemented")

    with open(f"dict/additional_tokens/{tokenizer_name}_new.txt", "a") as file:
        file.write(token + ",")

    tokenizer.add_to_vocab(token)

    e2w[token] = len(e2w)

    print("Added new token: ", token)
    return tokenizer, e2w


def make_remi_old():
    # I don't wanna be bothered with that
    pass


def make_cp_old():
    event2word = {"Bar": {}, "Position": {}, "Pitch": {}, "Duration": {}}
    word2event = {"Bar": {}, "Position": {}, "Pitch": {}, "Duration": {}}

    def special_tok(cnt, cls):
        """event2word[cls][cls+' <SOS>'] = cnt
        word2event[cls][cnt] = cls+' <SOS>'
        cnt += 1

        event2word[cls][cls+' <EOS>'] = cnt
        word2event[cls][cnt] = cls+' <EOS>'
        cnt += 1"""

        event2word[cls]["Pad_None"] = cnt
        word2event[cls][cnt] = "Pad_None"
        cnt += 1

        event2word[cls]["Mask_None"] = cnt
        word2event[cls][cnt] = "Mask_None"
        cnt += 1

    # Bar
    cnt, cls = 0, "Bar"
    event2word[cls]["Bar New"] = cnt
    word2event[cls][cnt] = "Bar New"
    cnt += 1

    event2word[cls]["Bar Continue"] = cnt
    word2event[cls][cnt] = "Bar Continue"
    cnt += 1
    special_tok(cnt, cls)

    # Position
    cnt, cls = 0, "Position"
    for i in range(1, 17):
        event2word[cls][f"Position {i}/16"] = cnt
        word2event[cls][cnt] = f"Position {i}/16"
        cnt += 1

    special_tok(cnt, cls)

    # Note On
    cnt, cls = 0, "Pitch"
    for i in range(22, 108):
        event2word[cls][f"Pitch {i}"] = cnt
        word2event[cls][cnt] = f"Pitch {i}"
        cnt += 1

    special_tok(cnt, cls)

    # Note Duration
    cnt, cls = 0, "Duration"
    for i in range(64):
        event2word[cls][f"Duration {i}"] = cnt
        word2event[cls][cnt] = f"Duration {i}"
        cnt += 1

    special_tok(cnt, cls)

    # print(event2word)
    # print(word2event)
    t = (event2word, word2event)

    with open("cp_old.pkl", "wb") as f:
        pickle.dump(t, f)


if __name__ == "__main__":
    config = miditok.TokenizerConfig(
        special_tokens=["PAD", "MASK"],
        pitch_range=(22, 108),
        beat_res={(0, 16): 4},
        num_velocities=127,
        use_programs=True,
        programs=(0, 1, 2, 3),
    )

    d = make_simple_new(config, "remi")
    with open("tsd.pkl", "wb") as f:
        pickle.dump(d, f)
    # make_cp_old()
