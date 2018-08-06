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
	source = "out/class2/02/score"
	parser.add_argument('-s', '--source_dir', default=source)
	parser.add_argument('-t', '--threshold', default=0)
	parser.add_argument('-g', '--global_step', default='016400')
	return parser.parse_args()

def pre_cmp(args):
	file = os.path.join(args.source_dir, 'dev-{}.json'.format(args.global_step))
	with open(file, 'r') as fh:
		data = json.load(fh)
	assert data['global_step']==int(args.global_step), "step not match"
	score, scorep, label = np.array(data['score']), np.array(data['scorep']), np.array(data['label'])
	scorepi = (scorep > float(args.threshold))
	common = scorepi & label
	print('precision: ', np.sum(common)/np.sum(label))
	print('recall: ', np.sum(common)/np.sum(scorepi))


if __name__ == '__main__':
	main()