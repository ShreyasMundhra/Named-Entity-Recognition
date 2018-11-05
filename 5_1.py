#!usr/bin/python
from utils import compute_transition_prob
from count_freqs import Hmm
import math
import os

def save_transition_probs(input_file):
	"""
	Computes and stores trigrams and their respective transition probabilities from an input file containing the trigrams
	"""

	# read counts file
	counter = Hmm(3)
	counter.read_counts(file('ner_rare.counts'))

	out_lines_list = []
	l = input_file.readline()
	while l:
		line = l.strip()
		if line:  # Nonempty line
			trigram = tuple(line.split())
			# get transition probability of trigram
			prob = compute_transition_prob(counter.ngram_counts[1][(trigram[0], trigram[1])], counter.ngram_counts[2][trigram])
			# get log probability
			log_prob = math.log(prob)
			l = line + " " + str(log_prob)

		out_lines_list.append(l)
		l = input_file.readline()
	out_lines = "\n".join(out_lines_list)

	# write trigrams and their log probs to file
	with open('5_1.txt','w') as out_file:
		out_file.write(out_lines)

if __name__ == "__main__":
	os.system('python 4_1.py')
	os.system('python count_freqs.py ner_train_rare.dat > ner_rare.counts')
	save_transition_probs(file('trigrams.txt'))