#!usr/bin/python
import math
import os
from count_freqs import Hmm
from utils import get_word_counts, compute_emission_probs, compute_transition_probs, dev_file_iterator, \
	get_sentences_in_dev_file
from algorithms import viterbi


def tagger(dev_file, transition_probs, emission_probs, freq_words, rare_symbol='_RARE_'):
	"""
	Computes tags and log probabilities using Viterbi algorithm and stores in file.
	"""
	tags = []
	probs = []

	# get iterator to iterate through all sentences in dev_file
	dev_file_sentence_iterator = get_sentences_in_dev_file(dev_file_iterator(dev_file))
	for sentence in dev_file_sentence_iterator:
		# replace rare or unseen words by rare_symbol
		for i in range(len(sentence)):
			if sentence[i] not in freq_words:
				sentence[i] = rare_symbol
		# get tags and probs {pi(k,u,v)} for the sentence using Viterbi algorithm
		sent_tags, sent_probs = viterbi(sentence, transition_probs, emission_probs)
		tags.append(sent_tags)
		probs.append(sent_probs)

	# write to file
	out_lines_list = []
	sentence_num = 0
	word_num = 0
	dev_file.seek(0)
	for word in dev_file_iterator(dev_file):
		# empty line
		if word is None:
			sentence_num = sentence_num + 1
			word_num = 0
			out_lines_list.append("")
			continue
		# non-empty line
		else:
			# previous tag is '*' in case of the first word of a sentence
			prev_tag = '*'
			if word_num > 0:
				prev_tag = tags[sentence_num][word_num - 1]
			curr_tag = tags[sentence_num][word_num]
			# get probability of the tagged sequence upto this word
			prob = probs[sentence_num][(word_num, prev_tag, curr_tag)]
			# get log probability
			log_prob = math.log(prob, 2)
			line = word + " " + tags[sentence_num][word_num] + " " + str(log_prob)
			word_num = word_num + 1
			out_lines_list.append(line)

	out_lines = "\n".join(out_lines_list)
	out_lines = out_lines + "\n"

	# write to file
	with open('5_2.txt','w') as out_file:
		out_file.write(out_lines)


if __name__ == "__main__":
	os.system('python 4_1.py')
	os.system('python count_freqs.py ner_train_rare.dat > ner_rare.counts')

	# get frequent words
	word_count_dict = get_word_counts(file('ner_train.dat'))
	freq_words = [word for word in word_count_dict if word_count_dict[word] >= 5]

	# get transition and emission probs
	counter = Hmm(3)
	counter.read_counts(file('ner_rare.counts'))
	transition_probs = compute_transition_probs(counter.ngram_counts[1], counter.ngram_counts[2])
	emission_probs = compute_emission_probs(counter.emission_counts, counter.ngram_counts[0])

	# store tagged data with the log probs to file
	tagger(file('ner_dev.dat'), transition_probs, emission_probs, freq_words)

	os.system('python eval_ne_tagger.py ner_dev.key 5_2.txt')