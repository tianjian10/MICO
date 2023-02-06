# Solutions for MICO

## Introduction

My solution for MICO is a metric based membership inference method, which is simple and less computational. I used the **per-sample hardness** to calibrate the score in different metrics, such as the **loss** and **confidence** of each sample. The [LiRA](https://arxiv.org/abs/2112.03570v1) is the most direct influence for my solution. The pipeline is described as follows,

1. Get the outputs of all samples with each model in the training set, calculate the loss and confidence and save them to advoid duplicate computation.
2. For each sample in the challenge set, collect the $scores_{in}$ and $scores_{out}$, and then calibrate the score in different ways.
3. (Unnecessary for some tracks)Merge the results from different scenarios.

As for shadow models, I only used the given 100 models in training set of each track and did not train any extra models. Take CIFAR-10 as an example, the ratio of training and test set is 5:1. That means for any one sample, about 5/6 of 100 models in the training set are IN models which were trained with that sample, and about 1/5 are OUT models.

As for the selection of calibration equations, I tried in different ways and found the best ones according to the results on validation set for different tracks and different scenarios at present.

## Dateset Preparation

In [src2](./) directory, you need prepare the models and data as follows,

```
data
├─cifar10
│  ├─cifar10_hi
│  ├─cifar10_inf
│  └─cifar10_lo
├─sst2
│  ├─sst2_hi
│  ├─sst2_inf
│  └─sst2_lo
├─purchase100
│  ├─purchase100_hi
│  ├─purchase100_inf
│  └─purchase100_lo
├─ddp
│  ├─cifar10_ddp
│  ├─purchase100_ddp
│  └─sst2_ddp
└─data
    ├─cifar10
    │  └─cifar-10-batches-py
    └─purchase100
```

## Getting Started

### CIFAR-10

```bash
# 1. get all outputs in parallel
python get_dataset.py -n 8 -j 1 -c cifar10

# 2. inference
# metric: loss formula: score - mu_in
python main.py -c cifar10 -m hardness -mi 0 -fi 1
# metric: confidence formula: score - mu_in
python main.py -c cifar10 -m hardness -mi 1 -fi 1

# 3. merge results
mkdir -p ./out_merge/cifar10
cp -r ./out_hardness_0_1/cifar10/cifar10_inf ./out_merge/cifar10
cp -r ./out_hardness_1_1/cifar10/cifar10_hi ./out_merge/cifar10
cp -r ./out_hardness_1_1/cifar10/cifar10_lo ./out_merge/cifar10
python main.py -c cifar10 --package
```

### Purchase-100

```bash
# 1. get all outputs in parallel
python get_dataset.py -n 8 -j 1 -c purchase100

# 2. inference
# metric: confidence formula: score - (mu_in + mu_out) / 2
python main.py -c purchase100 -m hardness -mi 1 -fi 0
# metric: loss formula: (score - mu_all) / sigma_all
python main.py -c purchase100 -m hardness -mi 0 -fi 2

# 3. merge results
mkdir -p ./out_merge/purchase100
cp -r ./out_hardness_1_0/purchase100/cifar10_inf ./out_merge/purchase100
cp -r ./out_hardness_0_2/purchase100/cifar10_hi ./out_merge/purchase100
cp -r ./out_hardness_0_2/purchase100/cifar10_lo ./out_merge/purchase100
python main.py -c purchase100 --package
```

### SST-2

```bash
# 1. get all outputs in parallel
python get_dataset.py -n 8 -j 1 -c sst2

# 2. inference
# metric: loss formula: score - mu_in
python main.py -c sst2 -m hardness -mi 0 -fi 1
```

### DP Distinguisher

```bash
# 1. get all outputs in parallel
python get_dataset.py -n 8 -j 1 -c ddp

# 2. inference
# metric: confidence formula: score - mu_in
python main.py -c ddp -m hardness -mi 1 -fi 1
```
