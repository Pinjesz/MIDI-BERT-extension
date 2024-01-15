import argparse
import numpy as np
import pickle
import os
from sklearn.metrics import confusion_matrix
from MidiBERT.common.cm_fig import save_cm_fig

from torch.utils.data import DataLoader
import torch
from transformers import BertConfig

from model import MidiBert
from finetune_trainer import FinetuneTrainer
from MidiBERT.common.finetune_dataset import FinetuneDataset
from finetune_model import TokenClassification, SequenceClassification

from prepare_data.new.cp.config import tokenization_dict


def get_args():
    parser = argparse.ArgumentParser(description="")

    ### tokenization ###
    parser.add_argument("--old", action="store_true")
    parser.add_argument("--tokenization", choices=["cp"], required=True)

    ### mode ###
    parser.add_argument(
        "--task", choices=["melody", "velocity", "composer", "emotion"], required=True
    )

    ### path setup ###
    parser.add_argument("--dict_file", type=str, default="dict/cp_old.pkl")
    parser.add_argument("--ckpt", type=str, default="")

    ### parameter setting ###
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--class_num", type=int)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="all sequences are padded to `max_seq_len`",
    )
    parser.add_argument("--hs", type=int, default=768)
    parser.add_argument("--index_layer", type=int, default=12, help="number of layers")
    parser.add_argument("--lr", type=float, default=2e-5, help="initial learning rate")

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default: false
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

    if args.old:
        args.dict_file = f"dict/cp_old.pkl"
    else:
        args.dict_file = f"dict/cp_new.pkl"

    root = "result/cp/finetune/"

    if args.task == "melody":
        args.class_num = 4
        args.ckpt = (
            root + "melody_default/model_best.ckpt" if args.ckpt == "" else args.ckpt
        )
    elif args.task == "velocity":
        args.class_num = 7
        args.ckpt = (
            root + "velocity_default/model_best.ckpt" if args.ckpt == "" else args.ckpt
        )
    elif args.task == "composer":
        args.class_num = 8
        args.ckpt = (
            root + "composer_default/model_best.ckpt" if args.ckpt == "" else args.ckpt
        )
    elif args.task == "emotion":
        args.class_num = 4
        args.ckpt = (
            root + "emotion_default/model_best.ckpt" if args.ckpt == "" else args.ckpt
        )

    return args


def load_data(dataset, task, is_old):
    data_root = f"Data/cp_old" if is_old else f"Data/cp_new"

    if dataset == "emotion":
        dataset = "emopia"

    if dataset not in ["pop909", "composer", "emopia"]:
        print(f"Dataset {dataset} not supported")
        exit(1)

    X_train = np.load(
        os.path.join(data_root, f"{dataset}_train.npy"), allow_pickle=True
    )
    X_val = np.load(os.path.join(data_root, f"{dataset}_valid.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(data_root, f"{dataset}_test.npy"), allow_pickle=True)

    print(
        "X_train: {}, X_valid: {}, X_test: {}".format(
            X_train.shape, X_val.shape, X_test.shape
        )
    )

    if dataset == "pop909":
        y_train = np.load(
            os.path.join(data_root, f"{dataset}_train_{task[:3]}ans.npy"),
            allow_pickle=True,
        )
        y_val = np.load(
            os.path.join(data_root, f"{dataset}_valid_{task[:3]}ans.npy"),
            allow_pickle=True,
        )
        y_test = np.load(
            os.path.join(data_root, f"{dataset}_test_{task[:3]}ans.npy"),
            allow_pickle=True,
        )
    else:
        y_train = np.load(
            os.path.join(data_root, f"{dataset}_train_ans.npy"), allow_pickle=True
        )
        y_val = np.load(
            os.path.join(data_root, f"{dataset}_valid_ans.npy"), allow_pickle=True
        )
        y_test = np.load(
            os.path.join(data_root, f"{dataset}_test_ans.npy"), allow_pickle=True
        )

    print(
        "y_train: {}, y_valid: {}, y_test: {}".format(
            y_train.shape, y_val.shape, y_test.shape
        )
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def conf_mat(_y, output, task, name):
    if task == "melody":
        target_names = ["M", "B", "A"]
        seq = False
    elif task == "velocity":
        target_names = ["pp", "p", "mp", "mf", "f", "ff"]
        seq = False
    elif task == "composer":
        target_names = [
            "Bethel",
            "Clayderman",
            "Einaudi",
            "Hancock",
            "Hillsong",
            "Hisaishi",
            "Ryuichi",
            "Yiruma",
        ]
        seq = True
    elif task == "emotion":
        target_names = ["HAHV", "HALV", "LALV", "LAHV"]
        seq = True

    output = output.detach().cpu().numpy()
    output = output.reshape(-1, 1)
    _y = _y.reshape(-1, 1)

    cm = confusion_matrix(_y, output)
    print(cm)

    _title = "BERT (CP): " + task + " task"

    save_cm_fig(cm, classes=target_names, normalize=False, title=_title, seq=seq, path=f'result/cp/eval/{name}/cm_cp_{task}.jpg')


def main():
    args = get_args()

    if args.use_wandb:
        import wandb

        config = {
            "type": "eval",
            "tokenization": args.tokenization,
            "task": args.task,
            "name": args.name,
            "max_seq_len": args.max_seq_len,
            "hidden state": args.hs,
            "index_layer": args.index_layer,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "cpu": args.cpu,
            "old": args.old
        }

        if args.old:
            wandb.init(
                project=args.project,
                job_type=f'eval_{args.task}_cp',
                config=config
            )
        else:
            for k, v in tokenization_dict.items():
                if type(v) is dict:
                    tokenization_dict[k] = f"{v}"

            config["tokenization_config"] = tokenization_dict

            wandb.init(
                project=args.project,
                job_type=f'eval_{args.task}_cp',
                config=config
                )

    print("Loading Dictionary")
    with open(args.dict_file, "rb") as f:
        e2w, w2e = pickle.load(f)

    print("\nBuilding BERT model")
    configuration = BertConfig(
        max_position_embeddings=args.max_seq_len,
        position_embedding_type="relative_key_query",
        hidden_size=args.hs,
    )

    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)

    print("\nLoading Dataset")
    if args.task == "melody" or args.task == "velocity":
        dataset = "pop909"
        model = TokenClassification(midibert, args.class_num, args.hs)
        seq_class = False
    elif args.task == "composer" or args.task == "emotion":
        dataset = args.task
        model = SequenceClassification(midibert, args.class_num, args.hs)
        seq_class = True

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset, args.task)

    trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val)
    testset = FinetuneDataset(X=X_test, y=y_test)

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(
        validset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print("   len of valid_loader", len(valid_loader))
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print("   len of test_loader", len(test_loader))

    print("\nLoad ckpt from", args.ckpt)
    best_mdl = args.ckpt
    checkpoint = torch.load(best_mdl, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    index_layer = int(args.index_layer) - 13
    print("\nCreating Finetune Trainer using index layer", index_layer)
    trainer = FinetuneTrainer(
        midibert,
        train_loader,
        valid_loader,
        test_loader,
        index_layer,
        args.lr,
        args.class_num,
        args.hs,
        y_test.shape,
        args.cpu,
        args.cuda_devices,
        model,
        seq_class,
    )

    test_loss, test_acc, all_output = trainer.test()
    print("test loss: {}, test_acc: {}".format(test_loss, test_acc))

    conf_mat(y_test, all_output, args.task, args.name)


if __name__ == "__main__":
    main()
