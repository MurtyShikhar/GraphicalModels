from factor import *
import copy
import math

def create_mrf(imgs, model, tables):
	'''
		returns a markov net corresponding to the images.
		A markov net is just an undirected graph. 
		Also returns all the corresponding factors
	'''
	assert(model in ['ocr','trans','skip','pair_skip'])
	flattened_img = [_id for img in imgs for _id in img]
	#print(flattened_img)
	# first, create a blank markov net
	l1 = len(imgs[0])
	l2 = len(flattened_img) 
	mrf = {i : set() for i, _ in  enumerate(flattened_img)}
	factors = set()

	charVocab = 10
	ocr_table , transition_table, skip_table ,pair_skip_table  = tables 

	for node_id in xrange(l2):
		#print(node_id)
		factors.add(Factor(set([node_id]), ocr_table[flattened_img[node_id]], {node_id: charVocab}))
		if model != 'ocr':
			# add an edge from node_id to node_id + 1 if not ocr model
			if (node_id != l1-1 and node_id != l2-1):
				mrf[node_id].add(node_id+1)
				mrf[node_id+1].add(node_id)
				phi = Factor(set([node_id, node_id+1]), transition_table, {node_id: charVocab, node_id+1: charVocab})
				factors.add(phi)

			# add skip edges
			if model != 'trans':
				if (node_id < l1):
					for i in xrange(0,node_id):
						if (imgs[0][i] == imgs[0][node_id] and i != node_id):
							mrf[node_id].add(i)
							mrf[i].add(node_id)
							phi = Factor(set([i, node_id]), skip_table, {node_id: charVocab, i : charVocab})
							factors.add(phi)
				else:
					for i in xrange(l1, node_id):
						if (imgs[1][i-l1] == imgs[1][node_id-l1] and i != node_id):
							mrf[node_id].add(i)
							mrf[i].add(node_id)
							phi = Factor(set([i, node_id]), skip_table, {node_id: charVocab, i : charVocab})
							factors.add(phi)


			# add pair skip edges
			if model == 'pair_skip':
				if (node_id < l1):
					for i in xrange(l1, l2):
						if (imgs[1][i-l1] == imgs[0][node_id] and i != node_id):
							mrf[node_id].add(i)
							mrf[i].add(node_id)
							phi = Factor(set([i, node_id]), pair_skip_table, {node_id: charVocab, i : charVocab})
							factors.add(phi)  	

	return mrf, factors



def selectNode(mrf):
	bestNode=None
	minFillEdges = 1<<20
	for node in mrf:
		newEdges=0
		for (n1, n2) in itertools.combinations(mrf[node],2):
			if (n2 not in mrf[n1]):
				newEdges+=1
			if newEdges > minFillEdges:
				break

		if (newEdges < minFillEdges):
			bestNode = node
			minFillEdges = newEdges

	return bestNode, minFillEdges

def compress(cliqueTree, topological_ordering):
	numNodes = len(topological_ordering)
	#print("performing compression")

	for i in xrange(numNodes):
		node = topological_ordering[i]
		for j in xrange(0, i):
			neighbor = topological_ordering[j]
			if neighbor in cliqueTree[node] and (neighbor.scope).issubset(node.scope):
				for neigbors2 in cliqueTree[neighbor]:
					cliqueTree[node].add(neigbors2)
					cliqueTree[neigbors2].remove(neighbor)
					cliqueTree[neigbors2].add(node)

				cliqueTree[node].remove(node)
				node.factors |= neighbor.factors
				del cliqueTree[neighbor]

	#print("clique tree:")
	#for node in cliqueTree:
	#	print(node)

	#print("-"*50)	
	return cliqueTree, topological_ordering

def min_fill_ordering(mrf, factors):
	'''
		returns a clique tree based on the min fill heuristic. (Naive implementation O(N^4))
		- mrf is just a dictionary of nodes with neigbors as value
		- factors is a set containing all factors
	'''

	#print(mrf)


	cliqueTree = {}
	numNodes = len(mrf)
	taus = {}
	topological_ordering = []

	for i in xrange(numNodes):
		currNode, minFillEdges = selectNode(mrf)
		#print("eliminating: %d with %d new edges" %(currNode, minFillEdges))
		# this is the scope of the clique
		cliqueScope = set([currNode])

		# all the neigbors of the currNode need to be connected
		neigbors = list(mrf[currNode])

		for i in xrange(len(neigbors)):
			n1 = neigbors[i]
			cliqueScope.add(n1)
			for j in xrange(i+1, len(neigbors)):
				n2 = neigbors[j]
				mrf[n1].add(n2)
				mrf[n2].add(n1)

			# eliminate this node as the neigbor
			mrf[n1].remove(currNode)


		# delete the node
		del mrf[currNode]

		factorsToRemove = []
		for factor in factors:
			if (factor.scope).issubset(cliqueScope):
		 		factorsToRemove.append(factor)

		for factor in factorsToRemove:
			factors.remove(factor)

		cliqueTreeNode = cliqueNode(cliqueScope, set(factorsToRemove))
		cliqueNodeNeigbors = set()

		tausToRemove = []
		for tau in taus:
			if tau.issubset(cliqueScope): # these nodes are neighbors
				cliqueNodeNeigbors.add(taus[tau])
				cliqueTree[taus[tau]].add(cliqueTreeNode)
				tausToRemove.append(tau)

		for tau in tausToRemove:
			del taus[tau]


		tau_curr = copy.deepcopy(cliqueScope)
		tau_curr.remove(currNode)
		tau_curr = frozenset(tau_curr)
		cliqueTree[cliqueTreeNode] = cliqueNodeNeigbors
		if len(tau_curr) != 0:
			taus[tau_curr] = cliqueTreeNode 

		topological_ordering.append(cliqueTreeNode)


	return compress(cliqueTree, topological_ordering[::-1])
	#return cliqueTree, topological_ordering




def get_tables(skip_strength=5.0, pair_skip_strength=5.0):
	f = open("OCRdataset-2/potentials/ocr.dat"); l = f.readlines(); f.close()
	l = map(lambda line: line.strip().split("\t"), l)
	chars = "d,o,i,r,a,h,t,n,s,e".split(",")
	char_inv = {char : val for (val, char) in enumerate(chars)}
	ocr_tables = [[0,0,0,0,0,0,0,0,0,0] for i in xrange(1000)]
	for (_id, char, prob) in l:
		ocr_tables[int(_id)][char_inv[char]] = math.log(float(prob))

	f = open("OCRdataset-2/potentials/trans.dat"); l = f.readlines(); f.close()
	l = map(lambda line: line.strip().split("\t"), l)
	trans_table = [0]*100
	for (c1, c2, prob) in l:
		trans_table[10*char_inv[c2] + char_inv[c1]] = math.log(float(prob))

	pair_skip_table = [0]*100
	skip_table = [0]*100
	for c1 in xrange(10):
		for c2 in xrange(10):
			if (c1==c2):
				pair_skip_table[10*c1+ c2] = math.log(pair_skip_strength)
				skip_table[10*c1+ c2] = math.log(skip_strength)
			else:
				pair_skip_table[10*c1 + c2] = 0.0
				skip_table[10*c1 + c2] = 0.0

	return ocr_tables, trans_table, skip_table, pair_skip_table


def get_data(img_dataset, truth_dataset):
	f = open("OCRdataset-2/data/%s" %img_dataset); l = f.readlines(); f.close()
	imgs = []
	curr = []
	for line in l:
		stripped_line = line.strip()
		if not stripped_line:
			imgs.append(curr)
			curr = []
		else:
			curr.append(map(int,stripped_line.split("\t")))

	f = open("OCRdataset-2/data/%s" %truth_dataset); l = f.readlines(); f.close()
	words = []
	curr = []
	for line in l:
		stripped_line = line.strip()
		if not stripped_line:
			words.append(curr)
			curr = []
		else:
			curr.append(stripped_line)


	return imgs, words


def get_stats(predictions, gold_labels, marginals):
	'''
		-predictions is a list of list of 2 words
		-gold labels is a list of list of 2 words
		-marginals is a list of dictionaries
	'''
	chars = "d,o,i,r,a,h,t,n,s,e".split(",")
	char_inv = {char : val for (val, char) in enumerate(chars)}

	flattened_predictions = [pred for preds in predictions for pred in preds]
	flattened_labels = [label for labels in gold_labels for label in labels]

	char_accuracy  = 0.0
	word_accuracy  = 0.0
	log_likelihood = 0.0
	totalChars=0

	for i, labels in enumerate(gold_labels):
		word = labels[0]+labels[1]
		for j, char in enumerate(word):
			log_likelihood += marginals[i][j][char_inv[char]]


	for i, _item in enumerate(zip(flattened_predictions, flattened_labels)):
		prediction, label = _item
		if label == prediction:
			word_accuracy += 1

		for c1, c2 in zip(prediction, label):
			totalChars +=1
			if c1==c2:
				char_accuracy += 1


		#print(prediction, label)

	print("word accuracy: %5.4f, char accuracy: %5.4f, log likelihood: %5.4f" %( (100*word_accuracy/len(flattened_labels)), (100*char_accuracy/totalChars), (log_likelihood/len(flattened_labels))))
