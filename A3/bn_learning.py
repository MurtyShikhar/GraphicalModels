from collections import defaultdict as ddict
import copy

class CPT:
	'''
		prior along with ordered set of domains
		evidence: ordered set of 
	'''
	def __init__(self, var,  vocab_size, evidence, evidenceVars=[]):
		self.var = var
		self.evidence = evidence
		self.scope = set(evidenceVars)
		self.scope.add(var)
		self.vocab_size = vocab_size
		self.evidenceVars = evidenceVars
		self.prior_count = ddict(float)
		if (len(evidence) != 0):
			self.table = {}
			for evidenceAssign in evidence:
				self.table[evidenceAssign] = ddict(float)


	def update(self, data):
		currAssignment = []
		for var in self.evidenceVars:
			currAssignment.append(data[var])

		if len(currAssignment):
			self.table[tuple(currAssignment)][data[self.var]] += 1.0

		self.prior_count[data[self.var]] += 1.0

	def normalize(self):
		if (len(self.evidenceVars) == 0):
			# this is a prior table
			total_sum = 0.0

			# go thru all the possible assignments to the CPT main variable
			# some of them might be zero, and that's where smoothing comes
			for entry in self.vocab_size:
				total_sum += (1.0 + self.prior_count[entry])

			# assignment to variables
			for entry in self.vocab_size:
				self.prior_count[entry] += 1.0
				self.prior_count[entry] /= total_sum

		else:
			# for all possible assignments to the evidence variables
			for evidenceAssign in self.table:
				total_sum = 0.0
				for entry in self.vocab_size:
					total_sum += (1.0 + self.table[evidenceAssign][entry])

				for entry in self.vocab_size:
					self.table[evidenceAssign][entry] += 1.0
					self.table[evidenceAssign][entry] /= total_sum


	def __str__(self):
		ret = '='*50
		ret += '\nvariable: %s\n' %(self.var)
		ret += 'possible: %s\n' %(self.vocab_size)
		if (len(self.evidenceVars) == 0):
			probabilities = [str(self.prior_count[entry]) for entry in self.vocab_size]
			zipped_vals = zip(self.vocab_size, probabilities)
			ret += "prior probabilities: %s" %zipped_vals

		else:
			ret += "evidence: %s\n" %(self.evidenceVars)
			for evidenceVar in self.table:
				probabilities = [self.table[evidenceVar][entry] for entry in self.vocab_size]
				zipped_vals = ["%s: %5.6f" %(var,probab) for var, probab in zip(self.vocab_size, probabilities)]
				ret += ", ".join(evidenceVar) + ": "
				ret += "{" +  ", ".join(zipped_vals) +  "}" + "\n"
		return ret

	def __repr__(self):
		return self.__str__()




def learn_bn_params(data, cpts):
	for dataPoint in data:
		for cpt in cpts:
			cpt.update(dataPoint)

	for cpt in cpts:
		cpt.normalize()


