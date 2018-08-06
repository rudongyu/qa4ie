import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict

def main():
	args = get_args()
	pro_data(args)

def get_args():
	parser = argparse.ArgumentParser()
	source = "out/final/00/score"
	parser.add_argument('-s', '--source_dir', default=source)
	parser.add_argument('-t', '--length', default=400)
	parser.add_argument('-g', '--global_step', default='150000')
	return parser.parse_args()

def pro_data(args):
	scorep, label, qaid, sentslen, rsent = [], [], [], [], []
	for part in range(4):
		print ('processing part {}...'.format(part))
		file = os.path.join(args.source_dir, 'dev-{}-{}.json'.format(args.global_step, part))
		with open(file, 'r') as fh:
			data = json.load(fh)
		assert data['global_step']==int(args.global_step), "step not match"
		scorep += data['scorep']
		label += data['label']
		qaid += data['id']
		sentslen += data['len']
		rsent += data['rsent']

	qadict = defaultdict(list)
	sentiddictp = {}
	sentiddict = {}
	total = 0
	recall = 0
	difnum = 0

	scorep, label, qaid, sentslen, rsent= np.array(scorep), np.array(label),\
		np.array(qaid), np.array(sentslen), np.array(rsent)
	sentid = rsent[:, -1]

	for i in tqdm(range(qaid.shape[0])):
		idxi = qaid[i]
		qadict[idxi].append((sentid[i], scorep[i], label[i], sentslen[i]))

	lengtht = int(args.length)
	for qaid, qainfo in zip(list(qadict.keys()), list(qadict.values())):
		sortedinfo = sorted(qainfo, key = lambda i: i[1], reverse = True)
		length_sum = 0
		i = 0
		while i < len(sortedinfo) and length_sum <= lengtht:
			length_sum += sortedinfo[i][3]
			i += 1
		real_label = [s[2] for s in sortedinfo]
		valid_label = [s[2] for s in sortedinfo[:i]]
		valid_sentid = [int(s[0]) for s in sortedinfo[:i]]
		sentiddict[qaid] = valid_sentid

		if(np.sum(real_label) > 0):
			difnum += 1
		if(np.sum(valid_label) == np.sum(real_label)):
			sentiddictp[qaid] = valid_sentid
		# elif(np.sum(valid_label) < np.sum(real_label)):
		# 	sentiddictm[qaid] = valid_sentid
		# else:
		# 	assert("something bad happened")

		recall += int(np.sum(valid_label)==np.sum(real_label))
		total += 1

	print (float(difnum)/total)
	print (float(recall)/total)
	with open("sentid+.json", "w") as fh:
		json.dump(sentiddictp, fh)
	with open("sentid-infer400-700.json", "w") as fh:
		json.dump(sentiddict, fh)

if __name__ == '__main__':
	main()