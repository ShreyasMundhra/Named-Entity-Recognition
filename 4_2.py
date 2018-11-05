#!usr/bin/python
from utils import get_word_counts, compute_emission_probs
from count_freqs import Hmm
import math
import os

def baseline_tagger(counts_file, dev_file, rare_symbol="_RARE_"):
	"""
	Implements a baseline tagger that uses only the emission probabilities to assign tags and stores in a file.
	"""

	# get frequently occurring words
	word_count_dict = get_word_counts(file('ner_train.dat'))
	freq_words = [word for word in word_count_dict if word_count_dict[word] >= 5]

	# compute emission probs
	counter = Hmm(3)
	counter.read_counts(counts_file)
	emission_probs = compute_emission_probs(counter.emission_counts, counter.ngram_counts[0])

	out_lines_list = []
	l = dev_file.readline()
	while l:
		word = l.strip()
		if word:  # Nonempty line
			# use emission probabilities of rare_symbol to assign tag and its probability for rare or unseen words.
			if word not in freq_words:
				tag = sorted(emission_probs[rare_symbol], key=emission_probs[word].get, reverse=True)[0]
				prob = emission_probs[rare_symbol][tag]

			# use emission probabilities of the word itself for frequently occurring words.
			else:
				tag = sorted(emission_probs[word], key=emission_probs[word].get, reverse=True)[0]
				prob = emission_probs[word][tag]
			log_prob = math.log(prob, 2)
			l = word + " " + tag + " " + str(log_prob)
		else:
			l = ""
		out_lines_list.append(l)
		l = dev_file.readline()
	out_lines = "\n".join(out_lines_list)
	out_lines = out_lines + "\n"

	# write words, corresponding tags and log probs to file
	with open('4_2.txt','w') as out_file:
		out_file.write(out_lines)

if __name__ == "__main__":
	os.system('python 4_1.py')
	os.system('python count_freqs.py ner_train_rare.dat > ner_rare.counts')
	baseline_tagger(file('ner_rare.counts'), file('ner_dev.dat'))
	os.system('python eval_ne_tagger.py ner_dev.key 4_2.txt')