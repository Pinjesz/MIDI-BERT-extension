import argparse
import numpy as np
import pickle
import os

from torch.utils.data import DataLoader
import torch
from transformers import BertConfig

from model import MidiBert
from finetune_trainer import FinetuneTrainer
from MidiBERT.common.finetune_dataset import FinetuneDataset

from prepare_data.new.remi.config import tokenization_dict as remi_dict
from prepare_data.new.tsd.config import tokenization_dict as tsd_dict
from prepare_data.new.structured.config import tokenization_dict as structured_dict


def get_args():
    parser = argparse.ArgumentParser(description='')

    ### tokenization ###
    parser.add_argument("--old", action="store_true")
    parser.add_argument("--tokenization", choices=["remi", "tsd", "structured"], required=True)

    ### mode ###
    parser.add_argument('--task', choices=['melody', 'velocity', 'composer', 'emotion'], required=True)
    ### path setup ###
    parser.add_argument('--dict_file', type=str)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--ckpt', type=str)

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument("--index_layer", type=int, default=12, help="number of layers")
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1,2,3], help="CUDA device ids")

    ### wandb ###
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument('--project', type=str, default='wimu')

    args = parser.parse_args()

    if args.old and args.tokenization != "remi":
        print("An 'old' tokenization is only available for remi and cp")
        exit(1)

    if args.old:
        args.dict_file = f"dict/{args.tokenization}_old.pkl"
    else:
        args.dict_file = f"dict/{args.tokenization}_new.pkl"

    args.ckpt = f'result/{args.tokenization}/pretrain/{args.name}/model_best.ckpt'

    if args.task == 'melody':
        args.class_num = 4
    elif args.task == 'velocity':
        args.class_num = 7
    elif args.task == 'composer':
        args.class_num = 8
    elif args.task == 'emotion':
        args.class_num = 4

    return args


def load_data(dataset, task, is_old, tokenization):
    data_root = f"Data/{tokenization}_old" if is_old else f"Data/{tokenization}_new"

    if dataset == 'emotion':
        dataset = 'emopia'

    if dataset not in ['pop909', 'composer', 'emopia']:
        print(f'Dataset {dataset} not supported')
        exit(1)

    X_train = np.load(os.path.join(data_root, f'{dataset}_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(data_root, f'{dataset}_valid.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(data_root, f'{dataset}_test.npy'), allow_pickle=True)

    print('X_train: {}, X_valid: {}, X_test: {}'.format(X_train.shape, X_val.shape, X_test.shape))

    if dataset == 'pop909':
        y_train = np.load(os.path.join(data_root, f'{dataset}_train_{task[:3]}ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_valid_{task[:3]}ans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_root, f'{dataset}_test_{task[:3]}ans.npy'), allow_pickle=True)
    else:
        y_train = np.load(os.path.join(data_root, f'{dataset}_train_ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_valid_ans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_root, f'{dataset}_test_ans.npy'), allow_pickle=True)

    print('y_train: {}, y_valid: {}, y_test: {}'.format(y_train.shape, y_val.shape, y_test.shape))

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    args = get_args()

    if args.tokenization == "remi":
        tokenization_dict = remi_dict
    elif args.tokenization == "tsd":
        tokenization_dict = tsd_dict
    elif args.tokenization == "structured":
        tokenization_dict = structured_dict

    if args.use_wandb:
        import wandb

        config = {
            "type": "finetune",
            "tokenization": args.tokenization,
            "task": args.task,
            "name": args.name,
            "max_seq_len": args.max_seq_len,
            "hidden state": args.hs,
            "index_layer": args.index_layer,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "max epochs": args.epochs,
            "lr": args.lr,
            "cpu": args.cpu,
            "old": args.old
        }

        if args.old:
            wandb.init(
                project=args.project,
                job_type=f'finetune_{args.task}_{args.tokenization}',
                config=config
            )
        else:
            for k, v in tokenization_dict.items():
                if type(v) is dict:
                    tokenization_dict[k] = f"{v}"

            config["tokenization_config"] = tokenization_dict

            wandb.init(
                project=args.project,
                job_type=f'finetune_{args.task}_{args.tokenization}',
                config=config
                )

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset")
    if args.task == 'melody' or args.task == 'velocity':
        dataset = 'pop909'
        seq_class = False
    elif args.task == 'composer':
        dataset = 'composer'
        seq_class = True
    elif args.task == 'emotion':
        dataset = 'emopia'
        seq_class = True
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset, args.task, args.old, args.tokenization)

    trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val)
    testset = FinetuneDataset(X=X_test, y=y_test)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader",len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader",len(valid_loader))
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader",len(test_loader))


    print("\nBuilding BERT model")
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                                position_embedding_type='relative_key_query',
                                hidden_size=args.hs)

    best_mdl = args.ckpt
    checkpoint = torch.load(best_mdl, map_location='cpu')
    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    midibert.load_state_dict(checkpoint['state_dict'], strict=False)


    index_layer = int(args.index_layer)-13
    print("\nCreating Finetune Trainer using index layer", index_layer)
    trainer = FinetuneTrainer(midibert, train_loader, valid_loader, test_loader, index_layer, args.lr, args.class_num,
                                args.hs, y_test.shape, args.cpu, args.cuda_devices, None, seq_class)


    print("\nTraining Start")
    save_dir = os.path.join(f'result/{args.tokenization}/finetune/', args.task + '_' + args.name)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    with open(os.path.join(save_dir, 'log'), 'a') as outfile:
        outfile.write("Loading pre-trained model from " + best_mdl.split('/')[-1] + '\n')
        for epoch in range(args.epochs):
            if bad_cnt >= 3:
                print('valid acc not improving for 3 epochs')
                break
            train_loss, train_acc = trainer.train()
            valid_loss, valid_acc = trainer.valid()
            test_loss, test_acc, _ = trainer.test()

            is_best = valid_acc > best_acc
            best_acc = max(valid_acc, best_acc)

            if is_best:
                bad_cnt, best_epoch = 0, epoch
            else:
                bad_cnt += 1

            print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {} | Test loss: {} | Test acc: {}'.format(
                epoch+1, args.epochs, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc))

            if args.use_wandb:
                wandb.log({
                    'epoch': epoch+1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'valid_loss': valid_loss,
                    'valid_acc': valid_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc
                })

            trainer.save_checkpoint(epoch, train_acc, valid_acc,
                                    valid_loss, train_loss, is_best, filename)


            outfile.write('Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_acc={}, valid_acc={}, test_acc={}\n'.format(
                epoch+1, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))


if __name__ == '__main__':
    main()
