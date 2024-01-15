import pickle
from miditok import TokenizerConfig
import numpy as np

from dict.make_dict import make_simple_new

DEFAULT_VELOCITY_BINS = np.array([0, 32, 48, 64, 80, 96, 128])

tokenizer = TokenizerConfig(
    special_tokens=["PAD", "MASK"],
    pitch_range=(22, 108),
    beat_res={(0, 16): 4},
    num_velocities=127,
    use_programs=True,
    programs=(0, 1, 2, 3),
    use_tempos=True,
)

tokenization_dict = tokenizer.to_dict()


Composer = {
    "Bethel": 0,
    "Clayderman": 1,
    "Einaudi": 2,
    "Hancock": 3,
    "Hillsong": 4,
    "Hisaishi": 5,
    "Ryuichi": 6,
    "Yiruma": 7,
    "Padding": 8,
}

Emotion = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
}


def main():
    config = TokenizerConfig.from_dict(tokenization_dict)
    _, vocabulary = make_simple_new(config, "structured")

    with open("dict/structured_new.pkl", "wb") as f:
        pickle.dump(vocabulary, f)


if __name__ == "__main__":
    main()
