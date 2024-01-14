import argparse
import numpy as np
import pickle
import os

from torch.utils.data import DataLoader
from transformers import BertConfig
from model import MidiBert
from trainer import BERTTrainer
from MidiBERT.common.midi_dataset import MidiDataset

from prepare_data.new.remi.config import config_dict as remi_dict
from prepare_data.new.tsd.config import config_dict as tsd_dict
from prepare_data.new.structured.config import config_dict as structured_dict


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--old", action="store_true")
    parser.add_argument("--tokenization", choices=["remi", "tsd", "structured"], required=True)

    ### path setup ###
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument('--dict_file', type=str)

    ### pre-train dataset ###
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["pop909", "composer", "pop1k7", "ASAP", "emopia"],
    )

    ### parameter setting ###
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument(
        "--mask_percent",
        type=float,
        default=0.15,
        help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="all sequences are padded to `max_seq_len`",
    )
    parser.add_argument("--hs", type=int, default=768)  # hidden state
    parser.add_argument(
        "--epochs", type=int, default=500, help="number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="initial learning rate")

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--cuda_devices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="CUDA device ids",
    )

    ### wandb ###
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--project", type=str, default="wimu")

    args = parser.parse_args()

    if args.old and args.tokenization != "remi":
        print("An 'old' tokenization is only available for remi and cp")
        exit(1)

    if args.old:
        args.dict_file = f"dict/{args.tokenization}_old.pkl"
    else:
        args.dict_file = f"dict/{args.tokenization}_new.pkl"

    return args


def load_data(datasets, is_old, tokenization):
    to_concat = []
    root = f"Data/{tokenization}_old" if is_old else f"Data/{tokenization}_new"

    for dataset in datasets:
        if dataset in {"pop909", "composer", "emopia"}:
            X_train = np.load(
                os.path.join(root, f"{dataset}_train.npy"), allow_pickle=True
            )
            X_valid = np.load(
                os.path.join(root, f"{dataset}_valid.npy"), allow_pickle=True
            )
            X_test = np.load(
                os.path.join(root, f"{dataset}_test.npy"), allow_pickle=True
            )
            data = np.concatenate((X_train, X_valid, X_test), axis=0)

        elif dataset == "pop1k7" or dataset == "ASAP":
            data = np.load(os.path.join(root, f"{dataset}.npy"), allow_pickle=True)

        print(f"   {dataset}: {data.shape}")
        to_concat.append(data)

    training_data = np.vstack(to_concat)
    print("   > all training data:", training_data.shape)

    # shuffle during training phase
    index = np.arange(len(training_data))
    np.random.shuffle(index)
    training_data = training_data[index]
    split = int(len(training_data) * 0.85)
    X_train, X_val = training_data[:split], training_data[split:]

    return X_train, X_val


def main():
    args = get_args()

    if args.tokenization == "remi":
        config_dict = remi_dict
    elif args.tokenization == "tsd":
        config_dict = tsd_dict
    elif args.tokenization == "structured":
        config_dict = structured_dict

    if args.use_wandb:
        import wandb

        if args.old:
            wandb.init(
                project=args.project,
                job_type=f"pretrain_{args.tokenization}",
                config={"type":"old"}
                )
        else:
            for k, v in config_dict.items():
                config_dict[k] = f"{v}"
            wandb.init(
                project=args.project,
                job_type=f"pretrain_{args.tokenization}",
                config=config_dict
                )

    print("Loading Dictionary")
    with open(args.dict_file, "rb") as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset")
    X_train, X_val = load_data(args.datasets, args.old, args.tokenization)

    trainset = MidiDataset(X=X_train)
    validset = MidiDataset(X=X_val)

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(
        validset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print("   len of valid_loader", len(valid_loader))

    print("\nBuilding BERT model")
    configuration = BertConfig(
        max_position_embeddings=args.max_seq_len,
        position_embedding_type="relative_key_query",
        hidden_size=args.hs,
    )

    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)

    print("\nCreating BERT Trainer")
    trainer = BERTTrainer(
        midibert,
        train_loader,
        valid_loader,
        args.lr,
        args.batch_size,
        args.max_seq_len,
        args.mask_percent,
        args.cpu,
        args.cuda_devices,
    )

    print("\nTraining Start")
    save_dir = f"result/{args.tokenization}/pretrain/" + args.name
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, "model.ckpt")
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    for epoch in range(args.epochs):
        if bad_cnt >= 30:
            print("valid acc not improving for 30 epochs")
            break
        train_loss, train_acc = trainer.train()
        valid_loss, valid_acc = trainer.valid()

        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)

        if is_best:
            bad_cnt, best_epoch = 0, epoch
        else:
            bad_cnt += 1

        print(
            "epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}".format(
                epoch + 1, args.epochs, train_loss, train_acc, valid_loss, valid_acc
            )
        )

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "valid_loss": valid_loss,
                    "valid_acc": valid_acc,
                }
            )

        trainer.save_checkpoint(
            epoch, train_acc, valid_acc, valid_loss, train_loss, is_best, filename
        )

        with open(os.path.join(save_dir, "log"), "a") as outfile:
            outfile.write(
                "Epoch {}: train_loss={}, train_acc={}, valid_loss={}, valid_acc={}\n".format(
                    epoch + 1, train_loss, train_acc, valid_loss, valid_acc
                )
            )


if __name__ == "__main__":
    main()
