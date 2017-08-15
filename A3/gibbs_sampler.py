from factor import *
from helpers import *
from collections import defaultdict as ddict
import random
import sys

def multiplyAll(factorList):
	if (len(factorList)==0):
		return None
	else:
		ret_factor = factorList[0]
		for i in xrange(1, len(factorList)):
			ret_factor = ret_factor.multiply(factorList[i])

		return ret_factor


def random_toss(node_factor):
	'''
		toss a coin, and binary search to find the right interval
	'''
	toss = random.random()

	lo = 0
	hi = len(node_factor.table)-1
	curr_val = 0.0

	for i in xrange(hi+1):
		if (toss < curr_val + math.exp(node_factor.table[i])):
			return i
		else:
			curr_val += math.exp(node_factor.table[i])

	return -1

def gibbs_sampler(tree, factors, query, vocab, num_itr=20000, burn_in=10000):
	'''
		query variables take on a fixed value while others will change
	'''

	if (len(query) != 0):
		factors = [factor.reduce(query) for factor in factors]

	# initial assignment to all variables
	curr_assignment = {}
	for node in tree:
		if node not in query:
			curr_assignment[node] = random.randint(0, len(vocab[node])-1)
		else:
			curr_assignment[node] = query[node] 

	nodeFactorMap = {node : [] for node in tree}
	counts = {node : ddict(int) for node in tree}

	for factor in factors:
		for node in factor.scope:
			nodeFactorMap[node].append(copy.deepcopy(factor))

	itr = 0
	samples = []

	while (itr < num_itr):
		for node in tree:
			if (node in query): continue
			del curr_assignment[node]
			node_factor = nodeFactorMap[node][0].reduce(curr_assignment)
			for i in xrange(1, len(nodeFactorMap[node])):
				node_factor = node_factor.multiply(nodeFactorMap[node][i].reduce(curr_assignment))

			node_factor.normalize()

			sample = random_toss(node_factor)
			curr_assignment[node] = sample

			if (itr > burn_in):
				samples.append(curr_assignment)
				for node in curr_assignment:
					counts[node][curr_assignment[node]] += 1.0

			itr+=1

	preds = {}
	for node in tree:
		if node in query:
			preds[node] = query[node]
		else:
			maxVal = -1
			pred = None
			for val in counts[node]:
				if (counts[node][val] > maxVal):
					maxVal = counts[node][val]
					pred = val
			preds[node] = pred


	return preds, counts, samples

