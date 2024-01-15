import argparse
import glob
import os
import pathlib
import pickle
import miditok
import miditoolkit

import numpy as np
from tqdm import tqdm
from dict.make_dict import make_simple_new, add_to_simple
from config import *


def get_args():
    parser = argparse.ArgumentParser(description="")
    ### mode ###
    parser.add_argument(
        "-t",
        "--task",
        default="",
        choices=["melody", "velocity", "composer", "emotion"],
    )

    ### path ###
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["pop909", "pop1k7", "ASAP", "pianist8", "emopia"],
    )
    parser.add_argument("--input_dir", type=str, default="")

    ### parameter ###
    parser.add_argument("--max_len", type=int, default=512)

    ### output ###
    parser.add_argument("--output_dir", default="Data/tsd_new")
    parser.add_argument(
        "--name", default=""
    )  # will be saved as "{output_dir}/{name}.npy"

    args = parser.parse_args()

    if args.task == "melody" and args.dataset != "pop909":
        print("[error] melody task is only supported for pop909 dataset")
        exit(1)
    elif args.task == "composer" and args.dataset != "pianist8":
        print("[error] composer task is only supported for pianist8 dataset")
        exit(1)
    elif args.task == "emotion" and args.dataset != "emopia":
        print("[error] emotion task is only supported for emopia dataset")
        exit(1)
    elif args.dataset == None and args.input_dir == None:
        print("[error] Please specify the input directory or dataset")
        exit(1)

    return args


def padding(data, max_len, e2w, ans=False):
    pad_len = max_len - len(data)
    for _ in range(pad_len):
        if not ans:
            data.append(e2w['Pad_None'])
        else:
            data.append(0)
    return data


class Event(miditok.Event):
    def __init__(self, event: miditok.Event, melody: int, velocity: int):
        super().__init__(event.type, event.value, event.time, event.program, event.desc)
        self.melody = melody
        self.velocity = velocity


def post_process(events: list[miditok.Event], e2w, task):
    result:list[Event] = []
    for i in range(len(events)):
        e = events[i]
        if e.type == "Program":
            if events[i + 1].type == "Pitch" and events[i + 2].type == "Velocity":
                vel = (
                    np.searchsorted(DEFAULT_VELOCITY_BINS, events[i + 2].value, side="right")
                    - 1
                )
                result.append(Event(events[i + 1], e.program, vel))
        elif e.type == "Pitch" or e.type ==  "Velocity":
            pass
        else:
            result.append(Event(e, 0, 0))

    ids = []
    ys = []
    for r in result:
        ids.append(e2w[str(r)])
        if task == "melody":
            ys.append(r.melody)
        if task == "velocity":
            ys.append(r.velocity)

    return ids, ys

def extract(files: list[str], args, tokenizer: miditok.MIDITokenizer, e2w, mode=""):
    assert len(files)
    print(f"number of {mode} files: {len(files)}")

    segments = []
    ans = []
    for path in tqdm(files):
        file = miditoolkit.MidiFile(path)

        success = False
        while not success:
            try:
                events = tokenizer.midi_to_tokens(file).events
                success = True
            except Exception as e:
                tokenizer, e2w = add_to_simple(tokenizer, e2w, e.args[0])

        tokens, ys = post_process(events, e2w, args.task)

        max_len = int(args.max_len)
        for i in range(0, len(tokens), max_len):
            segments.append(tokens[i : i + max_len])
            if args.task == "composer":
                name = path.split(os.sep)[-2]
                ans.append(Composer[name])
            elif args.task == "emotion":
                name = path.split(os.sep)[-1].split("_")[0]
                ans.append(Emotion[name])
            else:
                ans.append(ys[i : i + max_len])

        if len(segments[-1]) < max_len:
            if args.task == "composer" and len(segments[-1]) < max_len // 2:
                segments.pop()
                ans.pop()
            else:
                segments[-1] = padding(segments[-1], max_len, e2w, ans=False)
        if (args.task == "melody" or args.task == "velocity") and len(
            ans[-1]
        ) < max_len:
            ans[-1] = padding(ans[-1], max_len, e2w, ans=True)

    segments = np.array(segments)
    ans = np.array(ans)

    dataset = args.dataset if args.dataset != "pianist8" else "composer"

    if args.input_dir != "":
        if args.name == "":
            args.name = os.path.basename(os.path.normpath(args.input_dir))
        output_file = os.path.join(args.output_dir, f"{args.name}.npy")
    elif dataset == "composer" or dataset == "emopia" or dataset == "pop909":
        output_file = os.path.join(args.output_dir, f"{dataset}_{mode}.npy")
    elif dataset == "pop1k7" or dataset == "ASAP":
        output_file = os.path.join(args.output_dir, f"{dataset}.npy")

    np.save(output_file, segments)
    print(f"Data shape: {segments.shape}, saved at {output_file}")

    if args.task != "":
        if segments.shape[0] != ans.shape[0]:
            print(f"Answer shape: {ans.shape} does not fit data shape: {segments.shape}")
            exit(1)

        if args.task == "melody" or args.task == "velocity":
            ans_file = os.path.join(
                args.output_dir, f"{dataset}_{mode}_{args.task[:3]}ans.npy"
            )
        elif args.task == "composer" or args.task == "emotion":
            ans_file = os.path.join(args.output_dir, f"{dataset}_{mode}_ans.npy")
        np.save(ans_file, ans)
        print(f"Answer shape: {ans.shape}, saved at {ans_file}")


def main():
    args = get_args()
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # initialize model
    config = miditok.TokenizerConfig.from_dict(tokenization_dict)
    vocabulary = make_simple_new(config, "tsd")

    with open("dict/tsd_new.pkl", "wb") as f:
        pickle.dump(vocabulary, f)

    if args.dataset == "pop909":
        dataset = args.dataset
    elif args.dataset == "emopia":
        dataset = "EMOPIA_1.0"
    elif args.dataset == "pianist8":
        dataset = "joann8512-Pianist8-ab9f541"

    if args.dataset == "pop909" or args.dataset == "emopia":
        train_files = glob.glob(f"Data/Dataset/{dataset}/train/*.mid")
        valid_files = glob.glob(f"Data/Dataset/{dataset}/valid/*.mid")
        test_files = glob.glob(f"Data/Dataset/{dataset}/test/*.mid")

    elif args.dataset == "pianist8":
        train_files = glob.glob(f"Data/Dataset/{dataset}/train/*/*.mid")
        valid_files = glob.glob(f"Data/Dataset/{dataset}/valid/*/*.mid")
        test_files = glob.glob(f"Data/Dataset/{dataset}/test/*/*.mid")

    elif args.dataset == "pop1k7":
        files = glob.glob(f"Data/Dataset/{args.dataset}/midi_transcribed/*/*.midi")

    elif args.dataset == "ASAP":
        files = pickle.load(open("Data/Dataset/ASAP_song.pkl", "rb"))
        files = [f"Data/Dataset/asap-dataset/{file}" for file in files]

    elif args.input_dir:
        files = glob.glob(f"{args.input_dir}/*.mid")

    else:
        print("not supported")
        exit(1)

    if args.dataset in {"pop909", "emopia", "pianist8"}:
        extract(train_files, args, tokenizer, vocabulary[0], "train")
        extract(valid_files, args, tokenizer, vocabulary[0], "valid")
        extract(test_files, args, tokenizer, vocabulary[0], "test")
    else:
        # in one single file
        extract(files, args, tokenizer, vocabulary[0])


if __name__ == "__main__":
    main()
