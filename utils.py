#!usr/bin/python
import sys
from collections import defaultdict

from count_freqs import simple_conll_corpus_iterator

def get_word_counts(in_file):
	"""
	Get frequency of each word in in_file
	"""
	word_count_dict = defaultdict(int)

	line_iterator = simple_conll_corpus_iterator(in_file)
	for word, tag in line_iterator:
		if word is None:
			continue
		word_count_dict[word] += 1
	return word_count_dict

def compute_emission_probs(emission_counts, onegram_counts):
	"""
	Get emission probabilities
	"""
	emission_probs = defaultdict(defaultdict)
	for word, tag in emission_counts:
		# e(x|y) = Count(y->x)/Count(y)
		emission_probs[word][tag] = emission_counts[(word, tag)] / float(onegram_counts[tuple([tag])])
	return emission_probs

def compute_transition_prob(bigram_count, trigram_count):
	"""
	Get transition probability
	"""
	if bigram_count == 0:
		return 0
	return trigram_count / float(bigram_count)

def compute_transition_probs(bigram_counts, trigram_counts):
	"""
	Get transition probabilities of all valid trigrams
	"""
	transition_probs = defaultdict(float)
	for tag1 in ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'O']:
		for tag2 in ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC']:
			for tag3 in ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-PER', 'B-ORG', 'B-LOC', 'B-MISC', 'O', 'STOP']:
				bigram_count = bigram_counts[(tag1, tag2)]
				trigram_count = trigram_counts[(tag1, tag2, tag3)]
				transition_probs[(tag1, tag2, tag3)] = compute_transition_prob(bigram_count, trigram_count)

		for tag2 in ['O']:
			for tag3 in ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'O', 'STOP']:
				bigram_count = bigram_counts[(tag1, tag2)]
				trigram_count = trigram_counts[(tag1, tag2, tag3)]
				transition_probs[(tag1, tag2, tag3)] = compute_transition_prob(bigram_count, trigram_count)

	for tag1 in ['B-PER', 'B-ORG', 'B-LOC', 'B-MISC']:
		for tag2 in ['O']:
			for tag3 in ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'O', 'STOP']:
				bigram_count = bigram_counts[(tag1, tag2)]
				trigram_count = trigram_counts[(tag1, tag2, tag3)]
				transition_probs[(tag1, tag2, tag3)] = compute_transition_prob(bigram_count, trigram_count)

	for tag1 in ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-PER', 'B-ORG', 'B-LOC', 'B-MISC']:
		for tag2 in ['B-PER', 'B-ORG', 'B-LOC', 'B-MISC']:
			for tag3 in ['B-PER', 'B-ORG', 'B-LOC', 'B-MISC', 'O', 'STOP']:
				bigram_count = bigram_counts[(tag1, tag2)]
				trigram_count = trigram_counts[(tag1, tag2, tag3)]
				transition_probs[(tag1, tag2, tag3)] = compute_transition_prob(bigram_count, trigram_count)

	for tag1 in ['*']:
		for tag2 in ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'O']:
			for tag3 in ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-PER', 'B-ORG', 'B-LOC', 'B-MISC', 'O', 'STOP']:
				bigram_count = bigram_counts[(tag1, tag2)]
				trigram_count = trigram_counts[(tag1, tag2, tag3)]
				transition_probs[(tag1, tag2, tag3)] = compute_transition_prob(bigram_count, trigram_count)

		for tag2 in ['*']:
			for tag3 in ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'O', 'STOP']:
				bigram_count = bigram_counts[(tag1, tag2)]
				trigram_count = trigram_counts[(tag1, tag2, tag3)]
				transition_probs[(tag1, tag2, tag3)] = compute_transition_prob(bigram_count, trigram_count)

	return transition_probs

def dev_file_iterator(dev_file):
	"""
	Return iterator over the dev_file that yields one word at a time
	"""
	l = dev_file.readline()
	while l:
		word = l.strip()
		if word: # Nonempty line
			yield word
		else: # Empty line
			yield None
		l = dev_file.readline()

def get_sentences_in_dev_file(dev_file_iterator):
	"""
    Return an iterator object that yields one sentence at a time.
    """
	current_sentence = []  # Buffer for the current sentence
	for l in dev_file_iterator:
		if l == None:
			if current_sentence:  # Reached the end of a sentence
				yield current_sentence
				current_sentence = []  # Reset buffer
			else:  # Got empty input stream
				sys.stderr.write("WARNING: Got empty input file/stream.\n")
				raise StopIteration
		else:
			current_sentence.append(l)  # Add token to the buffer

	if current_sentence:  # If the last line was blank, we're done
		yield current_sentence  # Otherwise when there is no more token
	# in the stream return the last sentence.