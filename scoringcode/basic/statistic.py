import argparse
import json
import os
import re
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt


def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    file_size = "700-"
    source_dir = "/home/maxru/data/qa4iev3/{}".format(file_size)
    # source_dir = "/home/maxru/data/squad/"
    # target_dir = "data/squad/"
    target_dir = "stat"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-m', "--mode", default="seq", type=str)
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument('-f', "--file_size", default=file_size)
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    # TODO : put more args here
    return parser.parse_args()


def create_all(args):
    out_path = os.path.join(args.source_dir, "all-v1.1.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, "train-v1.1.json")
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, "dev-v1.1.json")
    dev_data = json.load(open(dev_path, 'r'))
    train_data['data'].extend(dev_data['data'])
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    if(args.mode=="seq"):
        # for i in range(5):
        #     prepro_each(args, 'train', 0.2*i, 0.2*(i+1), out_name='train'+str(i))
        # prepro_each(args, 'train', out_name='train')
        prepro_each(args, 'dev', out_name='dev')
        #prepro_each(args, 'test', out_name='itest')
        #prepro_each(args, "train", out_name="infer")
    if(args.mode=="squad"):
        prepro_each(args, 'train', out_name='train')
        prepro_each(args, 'dev', out_name='dev')

def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    else:
        raise Exception()

    # source_path = in_path or os.path.join(args.source_dir, "{}.seq.json".format(data_type))
    # source_data = json.load(open(source_path, 'r'))
    length_threshold = 150
    fh = open("noise_out{}".format(length_threshold), "w")
    lengthdict = defaultdict(lambda: 0)
    source_path = os.path.join(args.source_dir, "{}.seq.json".format(data_type))
    source_data = json.load(open(source_path, 'r'))
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            context = para['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            xi = list(map(word_tokenize, sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            xi = [[xijk for xijk in xij if xijk != ''] for xij in xi]
            for s in xi:
                if(len(s) > length_threshold    ):
                    fh.write(' '.join(s)+'\n')
            for l in list(map(len, xi)):
                lengthdict[l] += 1

    x, y = list(lengthdict.keys()), list(lengthdict.values())
    newx = [x[i] for i in range(len(x)) if y[i]>5]
    newy = [y[i] for i in range(len(x)) if y[i]>5]
    plt.bar(newx, newy)
    # plt.show()
    plt.savefig(os.path.join(args.target_dir, 'length-{}.png'.format(args.file_size)))
    fh.close()

            

def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens

if __name__ == "__main__":
    main()
