import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mico_competition import ChallengeDataset, load_model
from utils import Parallel, load_data

eps = 1.2e-38
data_dir = './data'
LEN_TRAINING = {'cifar10': 50000, 'purchase100': 150000, 'sst2': 67349}
NUM_CLASSES = {'cifar10': 10, 'purchase100': 100, 'sst2': 2}


class GetFeatures(object):

    def get_func(self, dataset_name):
        assert dataset_name in ['cifar10', 'purchase100', 'sst2']
        return getattr(self, f'_get_features_{dataset_name}', self._get_features_default)

    def _get_features_default(self, model, bx, by):
        out = model(bx)

        loss = -F.cross_entropy(out, by, reduction='none').view(-1, 1)

        prob = torch.softmax(out, dim=1)

        mask = F.one_hot(by, out.shape[1]).bool()
        conf = prob[mask].view(-1, 1)

        prob_y = prob.gather(1, by.view(-1, 1)).view(-1)
        mod_entropy = ((prob * (1 - prob + eps).log()).sum(dim=1) - prob_y * (1 - prob_y + eps).log() + (1 - prob_y) *
                       (prob_y + eps).log()).view(-1, 1)

        feat = torch.cat((out, loss, conf, mod_entropy), dim=1)

        return feat

    def _get_features_sst2(self, model, bx, by):
        out = model(**bx).logits

        loss = -F.cross_entropy(out, by, reduction='none').view(-1, 1)

        prob = torch.softmax(out, dim=1)

        mask = F.one_hot(by, out.shape[1]).bool()
        conf = prob[mask].view(-1, 1)

        prob_y = prob.gather(1, by.view(-1, 1)).view(-1)
        mod_entropy = ((prob * (1 - prob + eps).log()).sum(dim=1) - prob_y * (1 - prob_y + eps).log() + (1 - prob_y) *
                       (prob_y + eps).log()).view(-1, 1)

        feat = torch.cat((out, loss, conf, mod_entropy), dim=1)

        return feat


class Inference(object):

    def get_func(self, dataset_name):
        assert dataset_name in ['cifar10', 'purchase100', 'sst2']
        self.get_features = GetFeatures().get_func(dataset_name)
        if dataset_name == 'sst2':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        return getattr(self, f'_inference_{dataset_name}', self._inference_default)

    def _inference_default(self, model, loader):
        feats = []
        labels = []
        for bx, by in loader:
            bx = bx.cuda()
            by = by.cuda()
            feats.append(self.get_features(model, bx, by))
            labels.append(by)
        feats = torch.cat(feats)
        labels = torch.cat(labels)

        return feats.cpu(), labels.cpu()

    def _inference_sst2(self, model, loader):
        feats = []
        labels = []
        for batch in loader:
            bx = self.tokenizer(batch['sentence'], return_tensors='pt', padding='max_length', max_length=67)
            bx = bx.to(torch.device('cuda'))
            by = batch['label'].cuda()
            feats.append(self.get_features(model, bx, by))
            labels.append(by)
        feats = torch.cat(feats)
        labels = torch.cat(labels)

        return feats.cpu(), labels.cpu()


def get_dataset(model_folders, root, dataset, dataset_name, device):
    torch.cuda.set_device(device)
    train_x = []
    train_labels = []
    train_y = []

    batch_size = 1024
    inference = Inference().get_func(dataset_name)
    for model_idx, model_folder in enumerate(model_folders):
        start_time = time.perf_counter()

        path = os.path.join(root, model_folder)
        challenge_dataset = ChallengeDataset.from_path(path, dataset=dataset, len_training=LEN_TRAINING[dataset_name])
        member_indices = [challenge_dataset.challenge.indices[i] for i in challenge_dataset.member.indices
                          ] + [challenge_dataset.rest.indices[i] for i in challenge_dataset.training.indices]

        model = load_model(dataset_name, path)
        model.cuda().eval().requires_grad_(False)
        loader = DataLoader(dataset, batch_size=batch_size)
        x, labels = inference(model, loader)
        y = torch.zeros_like(labels)
        y[member_indices] = 1
        train_x.append(x)
        train_labels.append(labels)
        train_y.append(y)

        cost_time = time.perf_counter() - start_time
        print(f'[{model_idx}/{len(model_folders)}] folder: {model_folder} cost time: {cost_time:.2f}s')

    train_x = torch.stack(train_x)
    train_labels = torch.stack(train_labels)
    train_y = torch.stack(train_y)

    return train_x, train_labels, train_y


def collect_func(result_list):
    train_x = []
    train_labels = []
    train_y = []
    for result in result_list:
        result = result.get()
        train_x.append(result[0])
        train_labels.append(result[1])
        train_y.append(result[2])
    train_x = torch.cat(train_x)
    train_labels = torch.cat(train_labels)
    train_y = torch.cat(train_y)
    return train_x, train_labels, train_y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    seed = 0
    set_seed(seed)
    workers = Parallel(args.num_gpus, args.jobs_per_gpu)

    for scenario in args.scenarios:
        dataset_name = scenario.split('_')[0]
        data_file = f'./train_data_{scenario}.pt'
        if os.path.exists(data_file):
            print(f'load {data_file}')
            train_x, train_labels, train_y = torch.load(data_file)
        else:
            dataset = load_data(dataset_name)
            root = os.path.join(data_dir, args.challenge, scenario, 'train')
            model_folders = sorted(os.listdir(root), key=lambda d: int(d.split('_')[1]))
            # train_x shape: [100, n, dim]
            # train_labels shape: [100, n]
            # train_y shape: [100, n]
            train_x, train_labels, train_y = workers(func=get_dataset,
                                                     tasks=model_folders,
                                                     args=(root, dataset, dataset_name),
                                                     collect_func=collect_func)
            torch.save((train_x, train_labels, train_y), data_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_gpus', type=int, default=1)
    parser.add_argument('-j', '--jobs_per_gpu', type=int, default=1)
    parser.add_argument('-c',
                        '--challenge',
                        type=str,
                        default='cifar10',
                        choices=['cifar10', 'purchase100', 'sst2', 'ddp'])
    args = parser.parse_args()

    if args.challenge == 'ddp':
        args.scenarios = [f'cifar10_{args.challenge}', f'purchase100_{args.challenge}', f'sst2_{args.challenge}']
    else:
        args.scenarios = [f'{args.challenge}_inf', f'{args.challenge}_hi', f'{args.challenge}_lo']

    main(args)
