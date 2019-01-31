# -*- coding: utf-8 -*-
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from sklearn import metrics

def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def get_default_folder(dataset, feature):
    if dataset == "agnews":
        input = "input/ag_news_csv"
        output = "output/ag_news"
    elif dataset == "dbpedia":
        input = "input/dbpedia_csv"
        output = "output/dbpedia"
    elif dataset == "yelp_review":
        input = "input/yelp_review_full_csv"
        output = "output/yelp_review_full"
    elif dataset == "yelp_review_polarity":
        input = "input/yelp_review_polarity_csv"
        output = "output/yelp_review_polarity"
    elif dataset == "amazon_review":
        input = "input/amazon_review_full_csv"
        output = "output/amazon_review_full"
    elif dataset == "amazon_polarity":
        input = "input/amazon_review_polarity_csv"
        output = "output/amazon_review_polarity"
    elif dataset == "sogou_news":
        input = "input/sogou_news_csv"
        output = "output/sogou_news"
    elif dataset == "yahoo_answers":
        input = "input/yahoo_answers_csv"
        output = "output/yahoo_answers"
    return input, output + "_" + feature
