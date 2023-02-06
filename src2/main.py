import argparse
import csv
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from get_dataset import GetFeatures
from mico_competition import ChallengeDataset, load_model
from mico_competition.scoring import generate_roc, score
from utils import load_data

data_dir = './data'
eps = 1.2e-38
LEN_CHALLENGE = 100
LEN_TRAINING = {'cifar10': 50000, 'purchase100': 150000, 'sst2': 67349}
NUM_CLASSES = {'cifar10': 10, 'purchase100': 100, 'sst2': 2}


def get_hardness_pred(scores, masks, phase, model_idx, feat, num_classes, metric_idx, formula_idx):
    hardness_pred = []
    indices = range(feat.shape[1] - num_classes) if metric_idx == -1 else [metric_idx]

    for j in range(LEN_CHALLENGE * 2):
        for sel_idx in indices:
            sel_idx += num_classes
            # in scores
            in_mask = masks[:, j].clone()
            if phase == 'train':
                in_mask[model_idx] = False
            in_scores = scores[in_mask, j, sel_idx]

            # out scores
            out_mask = ~masks[:, j].clone()
            if phase == 'train':
                out_mask[model_idx] = False
            out_scores = scores[out_mask, j, sel_idx]

            # out scores' length may be zero in sst2
            if len(out_scores) == 0 or len(out_scores) == 1:
                out_scores = in_scores

            score_i = feat[j, sel_idx]
            in_mu = in_scores.mean()
            out_mu = out_scores.mean()
            all_mu = torch.cat([in_scores, out_scores]).mean()
            all_sigma = torch.cat([in_scores, out_scores]).std()

            if formula_idx == 0:
                hardness_pred.append(score_i - (in_mu + out_mu) / 2)
            elif formula_idx == 1:
                hardness_pred.append(score_i - in_mu)
            elif formula_idx == 2:
                hardness_pred.append((score_i - all_mu) / all_sigma)

    hardness_pred = torch.stack(hardness_pred).view(-1, len(indices))
    return hardness_pred


def inference(args):
    for scenario in args.scenarios:
        dataset_name = scenario.split('_')[0]

        # load data
        dataset = load_data(dataset_name)

        # preparation
        num_classes = NUM_CLASSES[dataset_name]
        if dataset_name == 'sst2':
            tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        train_x, train_labels, train_y = torch.load(f'./train_data_{scenario}.pt')

        # inference
        for phase in args.phases:
            root = os.path.join(data_dir, args.challenge, scenario, phase)
            model_folders = sorted(os.listdir(root), key=lambda d: int(d.split('_')[1]))
            predictions_list = []
            for model_idx, model_folder in enumerate(tqdm(model_folders, desc=f'{scenario} {phase} {args.method}')):
                path = os.path.join(root, model_folder)
                challenge_dataset = ChallengeDataset.from_path(path,
                                                               dataset=dataset,
                                                               len_training=LEN_TRAINING[dataset_name])
                challenge_points = challenge_dataset.get_challenges()

                if phase == 'train':
                    challenge_indices = challenge_dataset.challenge.indices
                    feat = train_x[model_idx, challenge_indices]
                    labels = train_labels[model_idx, challenge_indices]
                else:
                    # load model
                    model = load_model(dataset_name, path)
                    model.eval().requires_grad_(False).cuda()
                    challenge_dataloader = DataLoader(challenge_points, batch_size=2 * LEN_CHALLENGE)
                    get_feat = GetFeatures().get_func(dataset_name)
                    if dataset_name == 'sst2':
                        batch = next(iter(challenge_dataloader))
                        inputs = tokenizer(batch['sentence'], return_tensors='pt', padding='max_length', max_length=67)
                        labels = batch['label']
                    else:
                        inputs, labels = next(iter(challenge_dataloader))
                    inputs, labels = inputs.to(torch.device('cuda')), labels.cuda()
                    feat = get_feat(model, inputs, labels)
                if args.method == 'loss':
                    pred = feat[:, num_classes]
                elif args.method == 'confidence':
                    pred = feat[:, num_classes + 1]
                elif args.method == 'mod_entropy':
                    pred = feat[:, num_classes + 2]
                elif args.method == 'hardness':
                    challenge_indices = challenge_dataset.challenge.indices
                    scores = train_x[:, challenge_indices]
                    masks = train_y[:, challenge_indices].bool()
                    hardness_pred = get_hardness_pred(scores, masks, phase, model_idx, feat, num_classes,
                                                      args.metric_idx, args.formula_idx)
                    pred = hardness_pred[:, 0]

                predictions = pred.cpu().numpy()
                predictions_list.append(predictions)

            # min-max norm
            min_v = min([predictions.min() for predictions in predictions_list])
            max_v = max([predictions.max() for predictions in predictions_list])
            predictions_list = [(predictions - min_v) / (max_v - min_v) for predictions in predictions_list]

            # write to csv
            for idx, model_folder in enumerate(model_folders):
                root = os.path.join(args.out_dir, args.challenge, scenario, phase)
                path = os.path.join(root, model_folder)
                os.makedirs(path, exist_ok=True)
                predictions = predictions_list[idx]
                with open(os.path.join(path, 'prediction.csv'), 'w') as f:
                    csv.writer(f).writerow(predictions)


def scoring(args):
    all_scores = {}
    for scenario in args.scenarios:
        all_scores[scenario] = {}
        for phase in ['train']:
            predictions = []
            solutions = []
            root = os.path.join(data_dir, args.challenge, scenario, phase)
            out_root = os.path.join(args.out_dir, args.challenge, scenario, phase)
            model_folders = sorted(os.listdir(root), key=lambda d: int(d.split('_')[1]))
            for model_folder in tqdm(model_folders, desc=f'{scenario} {phase}'):
                path = os.path.join(root, model_folder)
                out_path = os.path.join(out_root, model_folder)
                predictions.append(np.loadtxt(os.path.join(out_path, 'prediction.csv'), delimiter=','))
                solutions.append(np.loadtxt(os.path.join(path, 'solution.csv'), delimiter=','))
            predictions = np.concatenate(predictions)
            solutions = np.concatenate(solutions)

            scores = score(solutions, predictions)
            all_scores[scenario][phase] = scores

    for scenario in args.scenarios:
        fpr = all_scores[scenario]['train']['fpr']
        tpr = all_scores[scenario]['train']['tpr']
        fig = generate_roc(fpr, tpr)
        fig.suptitle(f'{scenario}', x=-0.1, y=0.5)
        fig.tight_layout(pad=1.0)
        plt.savefig(f'{scenario}.png', bbox_inches='tight')
        plt.cla()

    final_score = []
    for scenario in args.scenarios:
        print(scenario)
        scores = all_scores[scenario]['train']
        scores.pop('fpr', None)
        scores.pop('tpr', None)
        scores.pop('TPR_FPR_500', None)
        scores.pop('TPR_FPR_1500', None)
        scores.pop('TPR_FPR_2000', None)
        print(pd.DataFrame([scores]).T.round(4))
        final_score.append(scores['TPR_FPR_1000'])
    print(f'TPR_FPR_1000: {np.round(final_score, 4)}')
    print(f'average TPR_FPR_1000: {np.mean(final_score):.4f}')
    with open(os.path.join(args.out_dir, f'{args.challenge}.txt'), 'w') as f:
        f.write(f'{args.out_dir} {args.challenge} {np.round(final_score, 4)} {np.mean(final_score):.4f}')


def package(args):
    os.chdir(args.out_dir)
    with zipfile.ZipFile(f'predictions_{args.challenge}.zip', 'w') as zipf:
        for scenario in args.scenarios:
            for phase in ['dev', 'final']:
                root = os.path.join(args.challenge, scenario, phase)
                model_folders = sorted(os.listdir(root), key=lambda d: int(d.split('_')[1]))
                for model_folder in tqdm(model_folders, desc=f'{scenario} {phase}'):
                    path = os.path.join(root, model_folder)
                    file = os.path.join(path, 'prediction.csv')
                    if os.path.exists(file):
                        zipf.write(file)
                    else:
                        raise FileNotFoundError(
                            f'`prediction.csv` not found in {path}. You need to provide predictions for all challenges')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--method',
                        type=str,
                        default='loss',
                        choices=[
                            'loss',
                            'confidence',
                            'mod_entropy',
                            'hardness',
                        ])
    parser.add_argument('-mi', '--metric_idx', type=int, default=-1)
    parser.add_argument('-fi', '--formula_idx', type=int, default=-1)
    parser.add_argument('-c',
                        '--challenge',
                        type=str,
                        default='cifar10',
                        choices=['cifar10', 'purchase100', 'sst2', 'ddp'])
    parser.add_argument('--package', action='store_true', default=False)
    args = parser.parse_args()

    if args.challenge == 'ddp':
        args.scenarios = [f'cifar10_{args.challenge}', f'purchase100_{args.challenge}', f'sst2_{args.challenge}']
    else:
        args.scenarios = [f'{args.challenge}_inf', f'{args.challenge}_hi', f'{args.challenge}_lo']

    args.phases = ['train', 'dev', 'final']

    if args.method == 'hardness':
        args.out_dir = f'./out_{args.method}_{args.metric_idx}_{args.formula_idx}'
    elif args.package:
        args.out_dir = './out_merge'
    else:
        args.out_dir = f'./out_{args.method}'

    if not args.package:
        inference(args)
    scoring(args)
    package(args)
