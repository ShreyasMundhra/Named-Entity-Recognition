#!usr/bin/python
from utils import get_word_counts, dev_file_iterator, get_sentences_in_dev_file, compute_transition_probs, compute_emission_probs
import math
from algorithms import viterbi
from count_freqs import Hmm
import os

# Numbers
# contains numbers
# caps and dots for abbreviations
# lower and dots
# All caps
# word combination

def get_category(rare_word):
	"""
	Returns the category of a rare or unseen word
	"""
	# word is a number
	if rare_word.isdigit():
		return '_NUMBER_'
	# word contains numbers
	elif any(char.isdigit() for char in rare_word):
		return '_SUBNUMBER_'
	# word is an abbreviation with just upper case letters and dots
	elif any(char.isupper() for char in rare_word) and any(char == '.' for char in rare_word) and all(
					char.isupper() or char == '.' for char in rare_word):
		return '_ABV_'
	# word just contains lower case letters and dots
	elif any(char.islower() for char in rare_word) and any(char == '.' for char in rare_word) and all(
					char.islower() or char == '.' for char in rare_word):
		return '_LOWER_AND_DOTS_'
	# all characters in the word are upper case
	elif all(char.isupper() for char in rare_word):
		return '_ALL_CAPS_'
	# all characters in the word are lower case
	elif all(char.islower() for char in rare_word):
		return '_ALL_LOWER_'
	# word is a combination of multiple words e.g. well-established
	elif any(char.isalpha() for char in rare_word) and any(char == '-' for char in rare_word) and all(
					char.isalpha() or char == '-' for char in rare_word):
		return '_WORD_COMB_'
	# word falls in none of the above categories
	else:
		return '_RARE_'

def tagger(dev_file, transition_probs, emission_probs, freq_words):
	"""
	Computes tags and log probabilities using Viterbi algorithm and stores in file.
	"""
	tags = []
	probs = []

	# get iterator to iterate through all sentences in dev_file
	dev_file_sentence_iterator = get_sentences_in_dev_file(dev_file_iterator(dev_file))
	for sentence in dev_file_sentence_iterator:
		# replace rare or unseen words by their category
		for i in range(len(sentence)):
			if sentence[i] not in freq_words:
				sentence[i] = get_category(sentence[i])
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
	with open('6.txt','w') as out_file:
		out_file.write(out_lines)

def replace_infrequent_words_with_categories(in_file, out_file, count_thresh=5):
	"""
	Replace words with frequency < count_thresh in in_file by their category and store in out_file.
	"""

	# get frequency of each word in in_file
	word_count_dict = get_word_counts(in_file)

	out_lines_list = []
	in_file.seek(0)
	l = in_file.readline()
	while l:
		line = l.strip()
		if line:  # Nonempty line
			fields = line.split(" ")
			word = " ".join(fields[:-1])

			# replace word with its category if frequency < count_thresh
			if word_count_dict[word] < count_thresh:
				line = " ".join([get_category(word), fields[-1]])
		out_lines_list.append(line)
		l = in_file.readline()
	out_lines = "\n".join(out_lines_list)
	out_file.write(out_lines)

if __name__ == "__main__":
	# replace infrequent words with categories and write to file
	replace_infrequent_words_with_categories(file('ner_train.dat'), file('ner_train_cats.dat', 'w'))

	# generate counts file
	os.system('python count_freqs.py ner_train_cats.dat > ner_cats.counts')

	# get frequent words
	word_count_dict = get_word_counts(file('ner_train.dat'))
	freq_words = [word for word in word_count_dict if word_count_dict[word] >= 5]

	# get transition and emission probabilities
	counter = Hmm(3)
	counter.read_counts(file('ner_cats.counts'))
	transition_probs = compute_transition_probs(counter.ngram_counts[1], counter.ngram_counts[2])
	emission_probs = compute_emission_probs(counter.emission_counts, counter.ngram_counts[0])

	# store tagged data with the log probs to file
	tagger(file('ner_dev.dat'), transition_probs, emission_probs, freq_words)

	os.system('python eval_ne_tagger.py ner_dev.key 6.txt')