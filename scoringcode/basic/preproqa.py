import argparse
import json
import os
import re
import numpy as np
from matplotlib import pyplot as plt
# data: q, cq, score, label
# shared: sents, csents, word_counter, char_counter, word2vec
# no metadata
from collections import Counter, defaultdict
from tqdm import tqdm


def main():
	args = get_args()
	prepro(args)


def get_args():
	parser = argparse.ArgumentParser()
	home = os.path.expanduser("~")
	file_size = "400-700"
	source_dir = "/home/maxru/data/qa4iev3/{}".format(file_size)
	# source_dir = "/home/maxru/data/squad/"
	# target_dir = "data/squad/"
	target_dir = "dataout/qa/{}".format(file_size)
	filter_file = "sentid+.json"
	glove_dir = os.path.join(home, "data", "glove")
	parser.add_argument("-f", "--filter_file", default=filter_file)
	parser.add_argument('-s', "--source_dir", default=source_dir)
	parser.add_argument('-t', "--target_dir", default=target_dir)
	parser.add_argument("--glove_corpus", default="6B")
	parser.add_argument("--glove_dir", default=glove_dir)
	parser.add_argument("--glove_vec_size", default=100, type=int)
	parser.add_argument("--tokenizer", default="PTB", type=str)
	parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
	parser.add_argument("--port", default=8000, type=int)
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
	# prepro_each(args, 'train', out_name='train')
	prepro_each(args, 'dev', out_name='dev')
	# prepro_each(args, 'test', out_name='test')
	# prepro_each(args, "train", out_name="infer")

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

	sent_tokenize0 = lambda para: [para]

	# source_path = in_path or os.path.join(args.source_dir, "{}.seq.json".format(data_type))
	# source_data = json.load(open(source_path, 'r'))

	total = 0
	debug_out = []
	debug_q = Counter()
	false_num = 0
	fnum = 0
	q, cq = [], []
	y = []
	sents, csents = [], []
	rsents, rcsents = [], []
	ids = []
	answerss = []
	q_counter, q_counter0 = {}, {}
	word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
	source_path = os.path.join(args.source_dir, "{}.seq.json".format(data_type))
	source_data = json.load(open(source_path, 'r'))
	filter_dict = json.load(open(args.filter_file, 'r'))
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
				context_sent_len.append(len_cur)
			assert len_cur-1 == len(context), (len_cur, len(context))

			# sentences in word-level
			sentsi = xi
			
			# sentences in char-level
			csentsi = [[list(xijk) for xijk in xij] for xij in xi]

			xp.append(sentsi)
			cxp.append(csentsi)

			for qa in para['qas']:
				# get question words
				qaid = qa["id"]
				q_counter[qa['question']] = q_counter.get(qa['question'], 0) + 1
				total += 1

				if(qaid in filter_dict):
					valid_sentid = sorted(filter_dict[qaid])
					inv_sentid = {k:v for v,k in enumerate(valid_sentid)}
					rsentsi = [ai, pi, valid_sentid]
					qi = word_tokenize(qa['question'])
					cqi = [list(qij) for qij in qi]
					# xi = list(map(word_tokenize, sent_tokenize(context)))
					newxi = [xi[sentid] for sentid in valid_sentid]
					word_num = list(map(len, newxi))
					newxi = [[x for s in newxi for x in s]]
					cnewxi = [[list(xijk) for xijk in xij] for xij in newxi]
					
					yi = []
					answers = []
					for answer in qa['answers']:
						yii = []
						answer_text = answer['text']
						ansi = word_tokenize(answer_text)
						answer_location = answer['answer_location']
						not_complete = False
						for ans_idx, answer_start in enumerate(answer_location):
							answer_stop = answer_start + len(ansi[ans_idx])
							answer_loc_senti = get_sent_loc_idx(context_sent_len, answer_start, answer_stop)
							if(answer_loc_senti not in valid_sentid):
								not_complete = True
								break
							start = sum(word_num[:inv_sentid[answer_loc_senti]])
							end = sum(word_num[:inv_sentid[answer_loc_senti]+1])
							try:
								pos = newxi[0].index(ansi[ans_idx], start, end)
							except:
								not_complete = True
								false_num +=1
								print(xi[answer_loc_senti], newxi[0][start-5:end+5], word_num, start, end, newxi[start:end], ansi)
								break
							yii.append(pos)
						if(not_complete):
							continue
						yi.append(yii)
						answers.append(answer_text)

					if(len(yi)==0):
						fnum += 1
						q_counter0[qa['question']] = q_counter0.get(qa['question'], 0) + 1
						continue

					for xij in newxi:
						for xijk in xij:
							word_counter[xijk] += 1
							lower_word_counter[xijk.lower()] += 1
							for xijkl in xijk:
								char_counter[xijkl] += 1

					for qij in qi:
						word_counter[qij] += 1
						lower_word_counter[qij.lower()] += 1
						for qijk in qij:
							char_counter[qijk] += 1

					q.append(qi)
					cq.append(cqi)
					y.append(yi)
					ids.append(qa['id'])
					rsents.append(rsentsi)
					rcsents.append(rsentsi)
					answerss.append(answers)

				if(qaid not in filter_dict):
					continue

	word2vec_dict = get_word2vec(args, word_counter)
	lower_word2vec_dict = get_word2vec(args, lower_word_counter)

	# add context here
	qx, qy = [], []
	print("{0}/{1}".format(len(q), total))
	for k in q_counter0.keys():
		if(float(q_counter0[k])/q_counter[k] > 0.05):
			qx.append(q_counter0[k])
			qy.append(q_counter[k])
			print(k, "{}/{}".format(q_counter0[k], q_counter[k]))

	xaxis = list(range(len(qx)))
	plt.bar(xaxis, qx, width=0.5)
	plt.bar(xaxis, qy, width=0.2)
	plt.show()
	data = {'q': q, 'cq': cq, '*x': rsents, '*cx': rcsents, 'y': y, "id": ids, "answer": answerss}
	shared = {'x':sents, 'cx':csents, 'word_counter': word_counter, 'char_counter': char_counter,
	 'lower_word_counter': lower_word_counter, 'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}
	# print(debug_q)
	# print("saving ...")
	# with open("debug_out.json", "w") as fh:
	#     json.dump(debug_out, fh)
	save(args, data, shared, out_name)

def get_label(loc_list, loc_size):
	label = [0]*(loc_size)
	for idx in loc_list:
		label[idx] = 1

	return label

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
