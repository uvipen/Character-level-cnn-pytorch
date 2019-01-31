# [PYTORCH] Character-level Convolutional Networks for Text Classification

## Introduction

Here is my pytorch implementation of the model described in the paper **Character-level Convolutional Networks for Text Classification** [paper](https://arxiv.org/abs/1509.01626). 

## Datasets:

Statistics of datasets I used for experiments. These datasets could be download from [link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)

| Dataset                | Classes | Train samples | Test samples |
|------------------------|:---------:|:---------------:|:--------------:|
| AG’s News              |    4    |    120 000    |     7 600    |
| Sogou News             |    5    |    450 000    |    60 000    |
| DBPedia                |    14   |    560 000    |    70 000    |
| Yelp Review Polarity   |    2    |    560 000    |    38 000    |
| Yelp Review Full       |    5    |    650 000    |    50 000    |
| Yahoo! Answers         |    10   |   1 400 000   |    60 000    |
| Amazon Review Full     |    5    |   3 000 000   |    650 000   |
| Amazon Review Polarity |    2    |   3 600 000   |    400 000   |

## Setting:

I almost keep default setting as described in the paper. For optimizer and learning rate, there are 2 settings I use:

- **SGD** optimizer with initial learning rate of 0.01. The learning rate is halved every 3 epochs.
- **Adam** optimizer with initial learning rate of 0.001.

Additionally, in the original model, one epoch is seen as a loop over batch_size x num_batch records (128x5000 or 128x10000 or 128x30000), so it means that there are records used more than once for 1 epoch. In my model, 1 epoch is a complete loop over the whole dataset, where each record is used exactly once.

## Training

If you want to train a model with common dataset and default parameters, you could run:
- **python train.py -d dataset_name**: For example, python train.py -d dbpedia

If you want to train a model with common dataset and your preference parameters, like optimizer and learning rate, you could run:
- **python train.py -d dataset_name -p optimizer_name -l learning_rate**: For example, python train.py -d dbpedia -p sgd -l 0.01

If you want to train a model with your own dataset, you need to specify the path to input and output folders:
- **python train.py -i path/to/input/folder -o path/to/output/folder**

You could find all trained models I have trained in [link](https://drive.google.com/open?id=1zzC4r0nn8yInWjCbVrVZPFYyOWJQizqh)

## Experiments:

I run experiments in 2 machines, one with NVIDIA TITAN X 12gb GPU and the other with NVIDIA quadro 6000 24gb GPU. For small and large models, you need about 1.6 gb GPU and 3.5 gb GPU respectively.

Results for test set are presented as follows:  A(B):
- **A** is accuracy reproduced here.
- **B** is accuracy reported in the paper.
I used SGD and Adam as optimizer, with different initial learning rate. You could find out specific configuration for each experiment in **output/datasetname_scale/logs.txt**, for example output/ag_news_small/logs.txt

Maximally, each experiment would be run for 20 epochs. Early stopping was applied with patience is set to 3 as default.

|      Size     |     Small  |     Large    |
|:---------------:|:--------------:|:--------------:|
|    ag_news    | 86.71(84.35) | 88.13(87.18) |
|   sogu_news   | 95.08(91.35) | 94.90(95.12) |
|    db_pedia   | 97.53(98.02) | 97.60(98.27) |
| yelp_polarity | 91.40(93.47) | 93.50(94.11) |
|  yelp_review  | 56.09(59.16) | 58.93(60.38) |
|  yahoo_answer | 65.91(70.16) | 64.93(70.45) |
| amazon_review | 56.77(59.47) | 59.01(58.69) |
|amazon_polarity| 92.54(94.50) | 93.85(94.49) |

The training/test loss/accuracy curves for each dataset's experiments (figures for small model are on the left side) are shown below:

- **ag_news**

<img src="visualization/char-cnn_small_agnews.png" width="420"> <img src="visualization/char-cnn_large_agnews.png" width="420"> 

- **sogou_news**

<img src="visualization/char-cnn_small_sogou_news.png" width="420"> <img src="visualization/char-cnn_large_sogou_news.png" width="420">

- **db_pedia**

<img src="visualization/char-cnn_small_dbpedia.png" width="420"> <img src="visualization/char-cnn_large_dbpedia.png" width="420">

- **yelp_polarity**

<img src="visualization/char-cnn_small_yelp_review_polarity.png" width="420"> <img src="visualization/char-cnn_large_yelp_review_polarity.png" width="420">

- **yelp_review**

<img src="visualization/char-cnn_small_yelp_review.png" width="420"> <img src="visualization/char-cnn_large_yelp_review.png" width="420">

- **yahoo! answers**

<img src="visualization/char-cnn_small_yahoo_answers.png" width="420"> <img src="visualization/char-cnn_large_yahoo_answers.png" width="420">

- **amazon_review**

<img src="visualization/char-cnn_small_amazon_review.png" width="420"> <img src="visualization/char-cnn_large_amazon_review.png" width="420">

- **amazon_polarity**

<img src="visualization/char-cnn_small_amazon_polarity.png" width="420"> <img src="visualization/char-cnn_large_amazon_polarity.png" width="420">

You could find detail log of each experiment containing loss, accuracy and confusion matrix at the end of each epoch in **output/datasetname_scale/logs.txt**, for example output/ag_news_small/logs.txt
