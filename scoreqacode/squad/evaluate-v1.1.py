""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    false = 0
    f1 = exact_match = total = 0
    fe = open('./em_result', 'w')
    for qaid, answer_text in zip(dataset["id"], dataset["answer"]):
        total += 1
        pred_text = predictions[qaid]
        f1 += metric_max_over_ground_truths(f1_score, pred_text, answer_text)
        em_res = metric_max_over_ground_truths(exact_match_score, pred_text, answer_text)
        exact_match += em_res
        print('%s:\t%d' % (qaid, em_res), file = fe)
    # for article in dataset:
    #     for paragraph in article['paragraphs']:
    #         for qa in paragraph['qas']:
    #             total += 1
    #             if qa['id'] not in predictions:
    #                 message = 'Unanswered question ' + qa['id'] + \
    #                           ' will receive score 0.'
    #                 # print(message, file=sys.stderr)
    #                 false += 1
    #                 total -= 1
    #                 continue
    #             ground_truths = list(map(lambda x: x['text'], qa['answers']))
    #             hasP = 0
    #             for str in ground_truths:
    #                 for i, ch in enumerate(str):
    #                     if ch in string.punctuation:
    #                         hasP = 1
    #             prediction = predictions[qa['id']]
    #             em_res = metric_max_over_ground_truths(
    #                 exact_match_score, prediction, ground_truths)
    #             print('%s:\t%d\t%d' % (qa['id'], em_res, hasP), file = fe)
    #             exact_match += em_res
    #             f1 += metric_max_over_ground_truths(
    #                 f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    fe.close()

    f = open('./result', 'a+')
    print('%.2f\t%.2f\t%d' % (exact_match, f1, total), file = f)
    f.close()

    return 1


if __name__ == '__main__':
    expected_version = '1.0'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        # dataset_json = json.load(dataset_file)
        # if (dataset_json['version'] != expected_version):
        #     print('Evaluation expects v-' + expected_version +
        #           ', but got dataset with v-' + dcataset_json['version'],
        #           file=sys.stderr)
        # dataset = dataset_json['data']
        dataset = json.load(dataset_file)
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    evaluate(dataset, predictions)