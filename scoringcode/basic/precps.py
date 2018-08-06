import numpy as np
import json
import os
import argparse
from tqdm import tqdm

def main():
	args = get_args()
	pre_cmp(args)

def get_args():
	parser = argparse.ArgumentParser()
	source = "out/class2/00/score"
	parser.add_argument('-s', '--source_dir', default=source)
	parser.add_argument('-t', '--threshold', default=800)
	parser.add_argument('-g', '--global_step', default='060000')
	return parser.parse_args()

def pre_cmp(args):
	file = os.path.join(args.source_dir, 'dev-{}.json'.format(args.global_step))
	dic = {}
	total = 0
	recall = 0
	with open(file, 'r') as fh:
		data = json.load(fh)
	assert data['global_step']==int(args.global_step), "step not match"
	print (data.keys())
	print (len(data['len']))
	score, scorep, label, qaid, sentslen = np.array(data['len']), \
		np.array(data['scorep']), np.array(data['label']), np.array(data['id']), np.array(data['len'])
	for i in tqdm(range(qaid.shape[0])):
		idi = qaid[i]
		if(dic.get(idi, None)==None):
			dic[idi] = [[scorep[i]], [label[i]], [sentslen[i]]]
		else:
			dic[idi][0].append(scorep[i])
			dic[idi][1].append(label[i])
			dic[idi][2].append(sentslen[i])
	lengtht = int(args.threshold)
	for _, qainfo in tqdm(enumerate(dic.values())):
		scorep, label, sentslen = np.array(qainfo[0]), np.array(qainfo[1]), np.array(qainfo[2])
		#print (np.sum(sentslen))
		start = 0.0
		end = 1.0
		# while(end-start>1e-9):
		# 	mid = start
		# 	scorepi = (scorep > mid)
		# 	newlength = np.sum(scorepi * sentslen)
		# 	if(newlength > lengtht):
		# 		start = mid
		# 	else:
		# 		end = mid
		for th in np.arange(0.0, 1.0, 0.001):
			scorepi = (scorep > th)
			newlength = np.sum(scorepi * sentslen)
			if(newlength < lengtht):
				break
			# common = scorepi & label
		common = scorepi & label
		if(np.sum(common)>0):
			recall += 1
		total += 1

	print (float(recall)/total)

	# scorepi = (scorep > float(args.threshold))
	# common = scorepi & label
	# print('precision: ', np.sum(common)/np.sum(label))
	# print('recall: ', np.sum(common)/np.sum(scorepi))


if __name__ == '__main__':
	main()