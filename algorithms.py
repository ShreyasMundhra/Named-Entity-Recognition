from collections import defaultdict

def viterbi(sentence, transition_probs, emission_probs):
	"""
	Implements Viterbi algorithm and returns the predicted tags and maximum probabilities {pi(k,u,v)} for sentence
	"""

	# init max probs
	max_probs = defaultdict(float)
	# max prob is 1 since sentence always starts with **
	max_probs[tuple([-1,'*','*'])] = 1.0
	# init back-pointer
	bp = defaultdict(str)

	n = len(sentence)

	# get valid tags for each position
	tag_sets = [{'*'}, {'*'}, {'I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'O'}]
	temp = [{'I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-PER', 'B-ORG', 'B-LOC', 'B-MISC', 'O'}] * (n - 1)
	tag_sets.extend(temp)

	for k in range(n):
		for u in tag_sets[k+1]:
			for v in tag_sets[k+2]:
				max_prob = 0
				tag = ""
				for w in tag_sets[k]:
					emission_prob = 0
					if v in emission_probs[sentence[k]].keys():
						emission_prob = emission_probs[sentence[k]][v]
					# get max prob given the tags (w,u,v) and the current word
					prob = max_probs[tuple([k-1,w,u])] * transition_probs[tuple([w,u,v])] * emission_prob
					if prob > max_prob:
						max_prob = prob
						tag = w
				# get max prob of the tag sequence till now given the current tag v and the previous tag u
				max_probs[tuple([k,u,v])] = max_prob
				bp[tuple([k,u,v])] = tag

	pred_tags = [""]*n
	max_prob = 0
	# get the last two tags that maximise the probability of the entire tagged sequence
	for u in tag_sets[n]:
		for v in tag_sets[n+1]:
			prob = max_probs[tuple([n-1,u,v])] * transition_probs[tuple([u,v,'STOP'])]
			if prob > max_prob:
				max_prob = prob
				pred_tags[n-2], pred_tags[n-1] = u, v

	# get the remaining tags using backpointers
	for k in range(n-3, -1, -1):
		pred_tags[k] = bp[tuple([k+2, pred_tags[k+1], pred_tags[k+2]])]

	return pred_tags, max_probs