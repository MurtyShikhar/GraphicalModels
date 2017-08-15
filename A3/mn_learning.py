from bn_learning import *
from factor import *
from gibbs_sampler import *
from learning_helpers import *
from collections import defaultdict as ddict
import math
import joblib
import sys

def convertCPTtoFactor(cpt, variables):
	'''
		takes a bayes net cpt and converts it to a markov net factor
	'''

	scope = cpt.scope
	# return a dictionary of variable : list pairs where the list contains the possible assignments to this variable
	scope_domain = {var : variables[var] for var in scope}
	#print(scope_domain)
	numPossibleValues = {val : len(scope_domain[val]) for val in scope}
	#print(numPossibleValues)
	factor = Factor(scope, [], numPossibleValues)


	# scope of the cpt
	table_size = reduce(operator.mul, numPossibleValues.values(), 1)
	cpt_table = []

	for i in xrange(table_size):
		# returns a list containing ids of the assignment
		assign = factor._assig(i)
		var_assignment = scope_domain[cpt.var][assign[cpt.var]]
		#print("var",var_assignment)
		evidence_vars  = tuple([scope_domain[var][assign[var]] for var in cpt.evidenceVars])
		#print("evidence",evidence_vars)
		if (len(cpt.evidence) != 0):
			entry = cpt.table[evidence_vars][var_assignment]
		else:
			entry = cpt.prior_count[var_assignment]			

		if entry==0:
			cpt_table.append(0)
		else:
			cpt_table.append(math.log(entry))

	factor.table = cpt_table
	return factor



def moralizeNet(tree, cpts):
	factors = set()
	for cpt in cpts:
		factors.add(convertCPTtoFactor(cpt))

	mrf = {var: set() for var in tree}

	# add an edge between every pair of edges 
	parent_pointers = {var: set() for var in tree}

	for var in tree:
		for child in tree[var]:
			parent_pointers[child].add(var)

	for var in tree:
		# connect all parents of var
		parents = list(parent_pointers[var])
		for i in xrange(len(parents)):
			for j in xrange(i):
				p1 = parents[i]
				p2 = parents[j]
				mrf[p1].add(p2)
				mrf[p2].add(p1)

		# undirected connection between var and all its children
		for child in tree[var]:
			mrf[child].add(var)
			mrf[var].add(child)


	return mrf, factors



######################################################### LEARNING ###########################################################

def consistent(data_point, current_assign, vocab):
	'''
		data_point: a dictionary of var, val pairs
		current_assign : a dictionary of var: val pairs
		check if currrent_assign is consistent with data_point
	'''
	for var in current_assign:
		if (data_point[var] != vocab[var][current_assign[var]]):
			return False

	return True

def learn_mn_params(mrf, factors, net_type, data, vocab, lr, reg, num_itr, early_stopping = False, validation_data = None):
	itr = 0

	if early_stopping:
		test_data, gold_labels, evidence_vars = validation_data

	#data_counts[i] = number of times feature "i" is on in the data
	data_counts = ddict(float)	
	factor_assignments = []
	num_features=0
	for factor in factors:
		curr_assigns = []
		for i in xrange(len(factor.table)):
			curr_assigns.append(factor._assig(i))
			num_features += 1

		factor_assignments.append(curr_assigns)


	data = data[0: min(100000, len(data))]
	for idx, dataPoint in enumerate(data):
		print "\r>> Done with %d/%d of data" %(idx+1, len(data)),
		sys.stdout.flush()
		for i, factor in enumerate(factors):
			for j in xrange(len(factor.table)):
				curr_assign = factor_assignments[i][j]
				if (consistent(dataPoint, curr_assign, vocab)):
					data_counts[(i,j)] += 1.0

	num_itr_gibbs = 4e7/num_features
	print("number of features: %d, number of samples for gibbs sampling: %d, reg = %d" %(num_features, num_itr_gibbs, reg))
	reg = reg/100000.0
	best_acc = -1.0
	while (itr < num_itr):
		_, counts, samples = gibbs_sampler(mrf, factors, {}, vocab, num_itr = num_itr_gibbs, burn_in = num_itr_gibbs/2)
		print(">> Done with %d/%d iterations of training" %(itr+1, num_itr))
		eps_stop = True
		for i, factor in enumerate(factors):
			for j in xrange(len(factor.table)):
				curr_assign = factor_assignments[i][j]
				count = 0.0
				for sample in samples:
					if (consistent(sample, curr_assign, vocab)):
						count+=1.0

				expected_count = count/float(len(samples))

				delta = (data_counts[(i,j)]/len(data)) - expected_count - 2*reg*factor.table[j]
				if (abs(delta) > 1e-5):
					eps_stop=False
				factor.table[j] = factor.table[j] + lr*delta
				
		if (early_stopping):
			acc = predict(test_data, gold_labels, evidence_vars, vocab, mrf, factors)

		if (acc > best_acc):
			print("================== FOUND BEST! ===============")
			print("acc = %5.4f" %acc)
			joblib.dump(factors,"%s_%d_mn.joblib" %(net_type,100000.0*reg))
			best_acc = acc


		if (eps_stop or itr == num_itr-1):
			print("gradient descent ran for %d iterations" %(itr+1))
			break

		
		itr += 1
	return 
