from mn_learning import *
from bn_learning import *
from gibbs_sampler import *
from learning_helpers import *
import joblib

from parser import *
import sys

##### hebar2 params: lr = 0.01, reg = 0.3


if __name__ == '__main__':
	if (len(sys.argv) < 6):
		print("USAGE: python learning_main.py <train> <arch> <net_type> <training_data> <test_data> <gold_labels>")
		sys.exit(1)


	arch = sys.argv[2]
	train = int(sys.argv[1])
	assert(arch in ['bayes', 'markov'])
	net_type = sys.argv[3]
	training_data = sys.argv[4]
	test_data = sys.argv[5]
	gold_labels = sys.argv[6]

	# load a pretrained model
	if (not train):
	
		if (arch == 'bayes'):
			tree, cpts, variables = parse("finals/%s" %net_type, True)
			gold_labels = getData("A3-data/%s" %gold_labels)
			evidence_vars, test_data = getTestData("A3-data/%s" %test_data, variables)

			factors = [convertCPTtoFactor(cpt, variables) for cpt in cpts]
			predict(test_data, gold_labels, evidence_vars, variables, tree, factors,"%s.bn.out" %net_type)

		else:
			tree, cpts, variables = parse("A3-data/%s" %net_type)
			gold_labels = getData("A3-data/%s" %gold_labels)
			evidence_vars, test_data = getTestData("A3-data/%s" %test_data, variables)
			factors = joblib.load("finals/%s_mn.joblib" %net_type)
			predict(test_data, gold_labels, evidence_vars, variables, tree, factors,"finals/%s.mn.out" %net_type)

		sys.exit(0)



	tree, cpts, variables = parse("A3-data/%s" %net_type)

	cptDict = {}
	for cpt in cpts:
		cptDict[cpt.var] = cpt

	data = getData("A3-data/%s" %training_data)
	gold_labels = getData("A3-data/%s" %gold_labels)
	evidence_vars, test_data = getTestData("A3-data/%s" %test_data, variables)


	if (arch == 'bayes'):
		learn_bn_params(data, cpts)
		writeNet("A3-data/%s" %net_type, cptDict)

		factors = [convertCPTtoFactor(cpt, variables) for cpt in cpts]


	else:
		factors = [convertCPTtoFactor(cpt, variables) for cpt in cpts]
		if (len(sys.argv) > 7):
			reg = int(sys.argv[7])
		else:
			reg = 1.0

		if (train == 2): #warm start
			factors = joblib.load("%s_%d_mn.joblib" %(net_type,reg))

		learn_mn_params(tree, factors,net_type, data, variables, lr = 0.5, reg = reg, num_itr = 20, early_stopping = True, validation_data = [test_data, gold_labels, evidence_vars])
		factors = joblib.load("%s_%d_mn.joblib" %(net_type,reg))
		writeMarkov("A3-data/%s" %net_type, factors)


	predict(test_data, gold_labels, evidence_vars, variables, tree, factors)
