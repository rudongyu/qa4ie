import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
	args = get_args()
	draw_pr(args)

def get_args():
	parser = argparse.ArgumentParser()
	source = "out/class2/00/score"
	parser.add_argument('-s', '--source_dir', default=source)
	parser.add_argument('-g', '--global_step', default='060000')
	return parser.parse_args()

def draw_pr(args):
	file = os.path.join(args.source_dir, 'infer-{}.json'.format(args.global_step))
	ps = []
	rs = []
	auc1 = 0
	auc2 = 0
	with open(file, 'r') as fh:
		data = json.load(fh)
	assert data['global_step']==int(args.global_step), "step not match"
	scorep, label = np.array(data['scorep']), np.array(data['label'])
	for threshold in np.arange(0.0, 1.0, 1./1000):
		scorepi = (scorep > threshold)
		common = scorepi & label
		try:
			r = np.sum(common)/np.sum(label)
		except:
			r = 0
		try:
			p = np.sum(common)/np.sum(scorepi)
		except:
			p = 0
			print (common, scorepi)
		auc1 += p*1./1000
		auc2 += r*1./1000
		ps.append(p)
		rs.append(r)

	th = np.arange(0.0, 1.0, 1./1000)
	print(auc1, auc2)
	# print (rs[-30:])
	# print (ps[-30:])

	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.plot(rs, th)
	plt.plot(rs, ps)
	plt.legend(['threshold', 'precision'])
	plt.savefig('pr-{}.png'.format(args.global_step))


if __name__ == '__main__':
	main()