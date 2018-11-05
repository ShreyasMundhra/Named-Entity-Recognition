#!usr/bin/python
from utils import get_word_counts

def replace_infrequent_words(in_file, out_file, count_thresh=5, symbol="_RARE_"):
	"""
	Replace words with frequency < count_thresh in in_file by symbol and store in out_file.
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

			# replace word with symbol if frequency < count_thresh
			if word_count_dict[word] < count_thresh:
				line = " ".join([symbol, fields[-1]])
		out_lines_list.append(line)
		l = in_file.readline()
	out_lines = "\n".join(out_lines_list)
	out_file.write(out_lines)

if __name__ == "__main__":
	replace_infrequent_words(file('ner_train.dat'), file('ner_train_rare.dat', 'w'), count_thresh=5, symbol="_RARE_")