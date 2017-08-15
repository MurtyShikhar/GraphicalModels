from gibbs_sampler import *
from factor import *
import math

def printFormat(factor, variables):
	sorted_scope = sorted(factor.scope)
	allvars = ( ", ".join(map(str, sorted_scope)))
	ret = "probability ( %s ) {\n" %allvars
	if (len(sorted_scope) == 1):
		vals = ", ".join(map(str, factor.table))
		ret += "  table %s;\n" %(vals)
		ret += "}\n"
		return ret

	for idx, val in enumerate(factor.table):
		currAssignment = factor._assig(idx)
		values = []
		for currVar in currAssignment:
			values.append(variables[currVar][currAssignment[currVar]])

		currAssignment = ", ".join(values)
		ret += "  (%s) %5.4f;\n" %(currAssignment, val)

	ret += "}\n"
	return ret


def getData(fileName):
	'''
		fileName: the training data
		returns: the training data in the form of a list of dictionaries. the dictionary contains variable,value pairs 
		where value is the particular value the variable takes
	'''
	f = open("%s.dat" %fileName)
	lines = f.readlines(); f.close()

	lines = map(lambda line: line.strip(), lines)
	idToVar = {i : var for i, var in enumerate(lines[0].split(" "))}

	data = []
	for i in xrange(1, min(100000,len(lines))):
		currLine = lines[i].split(" ")
		currData = {}
		for id in idToVar:
			currData[idToVar[id]] = currLine[id]

		data.append(currData)
	
	return data		

def getTestData(fileName, variables):
	'''
		fileName: the test data
		variables: a dictionary with var, list pair where the list contains the possible values the var can take

		returns: the partial query data in the form of a list of dictionaries. the list doesn't contain values but ids 
	'''
	f = open("%s.dat" %fileName)
	lines = f.readlines(); f.close()

	lines = map(lambda line: line.strip(), lines)
	idToVar = {i : var for i, var in enumerate(lines[0].split(" "))}
	varIdMap = {}
	for var in variables:
		idMap = {}
		for i, val in enumerate(variables[var]):
			idMap[val] = i

		varIdMap[var] = idMap

	data = []
	evidence_vars = []
	for i in xrange(1, len(lines)):
		currLine = lines[i].split(" ")
		currData = {}
		curr_evidence_vars = set()
		for id in idToVar:
			if (currLine[id] != '?'):
				currData[idToVar[id]] = varIdMap[idToVar[id]][currLine[id]]
				curr_evidence_vars.add(idToVar[id])

		data.append(currData)
		evidence_vars.append(curr_evidence_vars)

	return evidence_vars, data		

def print_predictions(fileName, variables, predictions):
	f = open("%s.dat" %fileName)
	lines = f.readlines(); f.close()

	lines = map(lambda line: line.strip(), lines)
	idToVar = {i : var for i, var in enumerate(lines[0].split(" "))}
	for pred in predictions:
		predicted = []
		for i in xrange(len(idToVar)):
			curr_var = idToVar[i]
			predictedVal = variables[curr_var][pred[curr_var]]
			predicted.append(predictedVal) 

		print(" ".join(predicted))



def predict(test_data, gold_labels, evidence_vars, variables, tree, factors,fileName=None):
	all_preds  = []
	all_counts = []

	for i, test_point in enumerate(test_data):
		preds, counts, _  = gibbs_sampler(tree, factors, test_point, variables)
		#print(preds)
		#print(evidence_vars[i])
		# for var in preds:
		# 	if var not in evidence_vars[i]:
		# 		print(counts[var])

		all_preds.append(preds)
		all_counts.append(counts)

	inv_variables = {}
	for var in variables:
		inv_variables[var] = {val: i for i,val in enumerate(variables[var])}


	if fileName:
		f = open(fileName, "w")
		for i, test_point in enumerate(test_data):
			curr_test = "test%d" %(i+1)
			curr_preds = all_preds[i]
			curr_evidence_vars = evidence_vars[i]

			#all variables
			numVars=1
			for var in curr_preds: 
				if var in curr_evidence_vars:
					continue

				f.write("%s-var%d " %(curr_test, numVars))
				curr_sum = sum(all_counts[i][var].itervalues())
				j = 1
				for val in inv_variables[var]:
					var_id = inv_variables[var][val]
					curr_probab = float(all_counts[i][var][var_id])/curr_sum
					f.write("%s-var%d-ass%d:%5.4f " %(curr_test, numVars, j, curr_probab))
					j+=1

				f.write("\n")
				numVars+=1
			f.write("\n")

		f.close()
	return get_stats_nets(all_preds, all_counts, gold_labels, evidence_vars, variables)



def get_stats_nets(all_preds, all_counts, gold_labels, evidence_vars, variables):
	accuracy = 0.0
	average_ll = 0.0
	total = 0.0

	inv_variables = {}
	for var in variables:
		inv_variables[var] = {val: i for i,val in enumerate(variables[var])}

	for i in xrange(len(all_preds)):

		ll = 0.0
		curr_preds  = all_preds[i]
		curr_counts = all_counts[i]
		curr_gold_labels = gold_labels[i]
		curr_evidence_vars = evidence_vars[i]

		for var in curr_preds:
			prediced = variables[var][curr_preds[var]]
			truth = curr_gold_labels[var]
			truth_id = inv_variables[var][truth]

			sum_count = 0.0
			for val in curr_counts[var]:
				sum_count += 1.0 + curr_counts[var][val]

			ll += (math.log(1.0 + curr_counts[var][truth_id]) - math.log(sum_count))

			if var not in curr_evidence_vars:
				if (prediced == truth): accuracy += 1.0
				total += 1.0

		average_ll += ll
	
	accuracy = 100.0*accuracy/total
	print("average accuracy: %5.4f" %(accuracy))
	print("average log likelihood: %5.4f" %(average_ll/len(all_preds)))
	return accuracy
