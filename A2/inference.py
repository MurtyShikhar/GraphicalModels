from factor import *
from helpers import *
from message_passing import *
from loopy_bp import *
import sys
import time

def check_calibration(belief_graph):
	pass

def predict(imgs, model, tables, alg, run_for, useMAP=False):
	assert(alg in ['mp', 'lbp'])
	chars = "d,o,i,r,a,h,t,n,s,e".split(",")
	char_inv = {char : val for (val, char) in enumerate(chars)}

	mrf, factors = create_mrf(imgs, model, tables)
	#print("done creating mrf.")

	if alg == 'mp':
		cliqueTree, ordering = min_fill_ordering(mrf, factors)
		begin_time = time.time()
		belief_graph = two_way_msg_passing(cliqueTree, useMAP)
	else:
		bethe_graph = create_bethe_cluser_graph(mrf, factors)
		begin_time = time.time()
		belief_graph  = loopy_bp(bethe_graph, 1.0e-10, run_for, useMAP)


	predictions = {}
	marginal_chars = {}
	for node in belief_graph:
		for var in node.scope:
			if var not in predictions:
				if useMAP:
					marginals = (node.belief).take_max(node.belief.scope - set([var]))					

				else:
					marginals = (node.belief).marginalize(node.belief.scope - set([var]))
				#print(marginals)
				predictions[var] = chars[max(enumerate(marginals.table), key = lambda x: x[1])[0]]
				probabs = [math.exp(val) for val in marginals.table]
				Z = math.log(sum(math.exp(val) for val in marginals.table))
				probabs = [val - Z for val in marginals.table]
				marginal_chars[var] = probabs

	word1 = ""
	word2 = ""

	l1 = len(imgs[0])
	l2 = len(imgs[1])
	for var in xrange(l1):
		word1 += predictions[var]

	for var in xrange(l1, l1+l2):
		word2 += predictions[var]

	return  [word1, word2], marginal_chars, time.time()- begin_time


def run(img_dataset, truth_dataset, model, alg, useMAP=False):
	all_imgs, gold_labels = get_data(img_dataset, truth_dataset)
	print("running %s algorithm on %s model with %s dataset" %(alg, model, img_dataset))

	predictions, marginals = [], []
	total_time = 0.0
	for imgs, label in zip(all_imgs, gold_labels):
		words, marginal_chars, time_taken = predict(imgs, model, tables, alg, 0.5, useMAP)
		predictions.append(words)
		marginals.append(marginal_chars)
		total_time += time_taken

	print("Total time: %5.4f" %total_time)
	get_stats(predictions, gold_labels, marginals)

if __name__ == '__main__':
	tables = get_tables()
	if (len(sys.argv) == 6):
		img_dataset = sys.argv[4]
		truth_dataset = sys.argv[5]
		model = sys.argv[1]
		alg  = sys.argv[2]
		useMAP = (sys.argv[3] == '1')
		assert(model in ['ocr', 'trans', 'skip', 'pair_skip'])
		run(img_dataset, truth_dataset, model, alg, useMAP)

	else:
		for img_dataset, truth_dataset in [('data-loops.dat', 'truth-loops.dat'), ('data-loopsWS.dat', 'truth-loopsWS.dat'), ('data-tree.dat', 'truth-tree.dat'),('data-treeWS.dat', 'truth-treeWS.dat')]:
			print('-'*100)
			for model in ['ocr', 'trans', 'skip', 'pair_skip']:
				run(img_dataset, truth_dataset, model, 'mp', True)
