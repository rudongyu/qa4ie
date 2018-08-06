import argparse
import json
import os
import re
import numpy as np
# data: q, cq, score, label
# shared: sents, csents, word_counter, char_counter, word2vec
# no metadata
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}

def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    file_size = "700-"
    source_dir = "/home/maxru/data/qa4iev3/"#{}".format(file_size)
    # source_dir = "/home/maxru/data/squad/"
    # target_dir = "data/squad/"
    target_dir = "data/scoring/"#{}".format(file_size)
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-m', "--mode", default="seq", type=str)
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--kernel_size", default=5, type=int)
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
        # prepro_each(args, "train", out_name="infer")
    if(args.mode=="squad"):
        prepro_each(args, 'train', out_name='train')
        prepro_each(args, 'dev', out_name='dev')

def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    # source_path = in_path or os.path.join(args.source_dir, "{}.seq.json".format(data_type))
    # source_data = json.load(open(source_path, 'r'))

    q, cq = [], []
    sents, csents = [], []
    rsents, rcsents = [], []
    sentslen = []
    labels = []
    ids = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    outfile = open('noise.txt', 'w')
    enum = 0
    total = 0
    overlap = 0
    if(args.mode=="squad"):
        source_path = os.path.join(args.source_dir, "{}.json".format(data_type))
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
                # context in sent-level
                contexti = sent_tokenize(context)
                context_sent_len = []
                len_cur = 0
                for cidx, c in enumerate(contexti):
                    len_cur += len(c) + 1
                    context_sent_len.append(len_cur)
                #assert len_cur-1 == len(context), (len_cur, len(context))

                # sentences in word-level
                sentsi = xi
                
                # sentences in char-level
                csentsi = [[list(xijk) for xijk in xij] for xij in xi]
                if args.debug:
                    print(sentsi)

                for xij in xi:
                    for xijk in xij:
                        word_counter[xijk] += len(para['qas'])
                        lower_word_counter[xijk.lower()] += len(para['qas'])
                        for xijkl in xijk:
                            char_counter[xijkl] += len(para['qas'])

                for qa in para['qas']:
                    # get question words
                    qaid = qa["id"]
                    qi = word_tokenize(qa['question'])
                    cqi = [list(qij) for qij in qi]
                    answer_loc_list = []
                    # if(len(qa['answers'])>1):
                    #     continue
                    answer = qa['answers'][0]
                    # for answer in qa['answers']:
                    answer_text = answer['text']
                    ansi = word_tokenize(answer_text)
                    answer_location = answer['answer_start']
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)
                    answer_loc_list.append(get_sent_loc_idx(context_sent_len, answer_start, answer_stop))
                    score = get_score(answer_loc_list, len(sentsi), args.kernel_size)
                    label = get_label(answer_loc_list, len(sentsi))
                    for si in range(len(sentsi)):
                        if(len(sentsi[si]) > 60 or noise_flag(sentsi[si])):
                            outfile.write(' '.join(sentsi[si])+'\n')
                            enum+=1
                            continue
                        sents.append([sentsi[si]])
                        sentslen.append(len(sentsi[si]))
                        csents.append([csentsi[si]])
                        q.append(qi)
                        cq.append(cqi)
                        scores.append(score[si])
                        labels.append(label[si])
                        ids.append(qaid)

                    for qij in qi:
                        word_counter[qij] += 1
                        lower_word_counter[qij.lower()] += 1
                        for qijk in qij:
                            char_counter[qijk] += 1

    else:
        fi = 0
        qlen = []
        slen = []
        for file_size in ['0-400', '400-700', '700-']:
            source_path = os.path.join(args.source_dir, "{0}/{1}.seq.json".format(file_size, data_type))
            source_data = json.load(open(source_path, 'r'))
            start_ai = int(round(len(source_data['data']) * start_ratio))
            stop_ai = int(round(len(source_data['data']) * stop_ratio))
            for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
                xp, cxp = [], []
                sents.append(xp)
                csents.append(cxp)
                for pi, para in enumerate(article['paragraphs']):
                    # wordss
                    context = para['context']
                    context = context.replace("''", '" ')
                    context = context.replace("``", '" ')
                    xi = list(map(word_tokenize, sent_tokenize(context)))
                    xi = [process_tokens(tokens) for tokens in xi]  # process tokens
                    xi = [[xijk for xijk in xij if xijk != ''] for xij in xi]
                    # context in sent-level
                    contexti = sent_tokenize(context)
                    context_sent_len = []
                    len_cur = 0
                    for cidx, c in enumerate(contexti):
                        len_cur += len(c) + 1
                        if(len(xi[cidx]) < 200):
                            slen.append(len(xi[cidx]))
                        context_sent_len.append(len_cur)
                    assert len_cur-1 == len(context), (len_cur, len(context))

                    # sentences in word-level
                    sentsi = xi
                    
                    # sentences in char-level
                    csentsi = [[list(xijk) for xijk in xij] for xij in xi]
                    xp.append([[sent] for sent in sentsi])
                    cxp.append([[csent] for csent in csentsi])

                    if args.debug:
                        print(sentsi)

                    for xij in xi:
                        for xijk in xij:
                            word_counter[xijk] += len(para['qas'])
                            lower_word_counter[xijk.lower()] += len(para['qas'])
                            for xijkl in xijk:
                                char_counter[xijkl] += len(para['qas'])

                    for qa in para['qas']:
                        # get question words
                        total += 1
                        qaid = qa["id"]
                        qi = word_tokenize(qa['question'])
                        for qw in qi:
                            oflag = False
                            for xs in xi[0]:
                                if qw not in STOPWORDS and qw in xs:
                                    overlap += 1
                                    oflag = True
                                    break
                            if(oflag):
                                break
                        qlen.append(len(qi))
                        cqi = [list(qij) for qij in qi]
                        answer_loc_list = []
                        # if(len(qa['answers'])>1):
                        #     continue
                        answer = qa['answers'][0]
                        # for answer in qa['answers']:
                        answer_text = answer['text']
                        ansi = word_tokenize(answer_text)
                        answer_location = answer['answer_location']
                        api = []
                        for ans_idx, answer_start in enumerate(answer_location):
                            answer_stop = answer_start + len(ansi[ans_idx])
                            answer_loc_senti = get_sent_loc_idx(context_sent_len, answer_start, answer_stop)
                            answer_loc_list.append(answer_loc_senti)
                        label = get_label(answer_loc_list, len(sentsi))
                        for si in range(len(sentsi)):
                            if(len(sentsi[si]) > 60 or noise_flag(sentsi[si])):
                                outfile.write(' '.join(sentsi[si])+'\n')
                                enum+=1
                                continue
                            rsentsi = [ai+fi, pi, si]
                            rx = rsentsi
                            assert(sentsi[si] == sents[rx[0]][rx[1]][rx[2]][0])
                            #sents.append([sentsi[si]])
                            sentslen.append(len(sentsi[si]))
                            #csents.append([csentsi[si]])
                            q.append(qi)
                            cq.append(cqi)
                            labels.append(label[si])
                            ids.append(qaid)
                            rsents.append(rsentsi)
                            rcsents.append(rsentsi)

                        for qij in qi:
                            word_counter[qij] += 1
                            lower_word_counter[qij.lower()] += 1
                            for qijk in qij:
                                char_counter[qijk] += 1

                if args.debug:
                    break

            fi += stop_ai-start_ai

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    print(len(q), len(cq), len(labels))
    print(float(overlap)/total)
    print(enum)
    data = {'q': q, 'cq': cq, '*sents': rsents, '*csents': rcsents, 'label': labels, "id": ids, 
        "sentslen": sentslen}
    shared = {'sents': sents, 'csents': csents, 'word_counter': word_counter, 'char_counter': char_counter,
     'lower_word_counter': lower_word_counter, 'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    # print("saving ...")
    # save(args, data, shared, out_name)

    plt.figure()
    sns.set( palette="muted", color_codes=True)  
    sns.distplot(qlen, kde_kws={"label":"Question Length Distribution"})
    plt.savefig("qld")
    plt.figure()
    sns.distplot(slen, kde_kws={"label":"Sentence Length Distribution"})
    plt.savefig("sld")
    plt.show()


def get_score(loc_list, loc_size, kernel_size):
    score = [0]*(loc_size + kernel_size - 1)
    res = []
    pad = int((kernel_size-1)/2)
    kernel = gaussian_kernel(kernel_size)

    for idx in loc_list:
        score[idx+pad] = 1
    for idx in range(pad):
        score[idx] = score[pad]
    for idx in range(-pad, 0):
        score[idx] = score[-pad-1]

    # gaussian smoothing
    for i in range(loc_size):
        res.append(np.sum(kernel*np.array(score[i:i+kernel_size])))
    
    return list(np.array(res)/max(res))

def get_label(loc_list, loc_size):
    label = [0]*(loc_size)
    for idx in loc_list:
        label[idx] = 1

    return label

def gaussian_kernel(kernel_size, sigma = 1):
    pad = (kernel_size - 1)/2
    return 1.0/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.square(np.array(range(kernel_size)) - pad)/(2*sigma**2))

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

def get_sent_loc_idx(list_len, start, end):
    i = 0
    while(start >= list_len[i]):
        i += 1
    return i

def noise_flag(sent_list):
    m = 0
    n = 0
    for idx, word in enumerate(sent_list):
        if (':' in word):
            m+=1
        if ('-' in word):
            n+=1
        if(m>10 or n > 10):
            return True
    return False


if __name__ == "__main__":
    main()
