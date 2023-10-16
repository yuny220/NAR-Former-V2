# NAR-Former V2: Rethinking Transformer for Universal Neural Network Representation Learning

This is the source code for paper:<br> 
[NAR-Former V2: Rethinking Transformer for Universal Neural Network Representation Learning](https://arxiv.org/abs/2306.10792)

## The link to the released files
[NAR-Former V2](https://drive.google.com/drive/folders/1JtErGuWwPoR9lobxmT_u2BHI57hSAnLV?usp=sharing)

## Experiments of latency prediction on NNLQP
Here is the guide to train and test our NAR-Former V2 model for latency prediction on the NNLQP dataset.

### Data preparation
Download the `unseen_structure` part of [NNLQP](https://github.com/ModelTC/NNLQP) and put it in `dataset/`. Download the `dataset/unseen_structure/gt_stage.txt` and put it in `dataset/unseen_structure/`.

### Train NAR-Former V2
You can directly download the `experiments/latency_prediction/in_domain/checkpoints/ckpt_best.pth` or train from scratch following the steps below:
Change the `BASE_DIR` in `experiments/latency_prediction/in_domain/train.sh` to the absolute path of our codes and run:

```
cd experiments/latency_prediction/in_domain/
bash train.sh
```

The pretrained models will be saved in `experiments/latency_prediction/in_domain/checkpoints/`.

### Test NAR-Former V2

Change the `BASE_DIR` in `experiments/latency_prediction/in_domain/test.sh` to the absolute path of our codes and run:
```
cd experiments/latency_prediction/in_domain/
bash test.sh
```

## Experiments of accuracy prediction on NAS-Bench-201
Here is the guide to train and test our NAR-Former V2 model for accuracy prediction on the NAS-Bench-201 dataset.

### Data preparation
Download the preprocessed data file `dataset/nasbench201/all.pt`.

### Train NAR-Former V2
You can directly download the `experiments/accuracy_prediction/nasbench201/checkpoints/ckpt_best.pth` and or train from scratch following the steps below:
Change the `BASE_DIR` in `experiments/accuracy_prediction/nasbench201/train.sh` to the absolute path of our codes and run:
```
cd experiments/accuracy_prediction/nasbench201/
bash train.sh
```
The pretrained models will be saved in `experiments/accuracy_prediction/nasbench201/checkpoints/`.

### Test NAR-Former V2
Change the `BASE_DIR` in `experiments/accuracy_prediction/nasbench201/test.sh` to the absolute path of our codes and run:
```
cd experiments/accuracy_prediction/nasbench201/
bash test.sh
```
## The example organization of dataset folder

|__dataset

    |--unseen_structure

        |--gt.txt

        |--gt_stage.txt

        |__onnx

            |--...

            |--...

    |--nasbench101

        |__all.pt

    |__nasbench201

        |__all.pt

## Acknowledge
1. [NAS-Bench-101](https://github.com/google-research/nasbench)
2. [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201)
3. [NNLQP](https://github.com/auroua/NPENASv1)