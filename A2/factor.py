from collections import defaultdict as ddict
import random
import operator, copy
import itertools
import math
'''
	The phis in a markov net. 
	Also has functionality for marginalisation, maximisation etc
'''
class Factor:
	def __init__(self, _scope = set(), _table = [], _numPossibleValues = {}):
		'''
			- scope is a set
			- table is a flattened out distribution table
			- numPossibleValues is a dictionary containing the cardinality of each variable in scope
		'''
		self.scope = _scope
		
		self.table = _table
		self.numPossibleValues = _numPossibleValues
		self.strides = ddict(int)

		running_product =1
		ordered_scope = sorted(self.scope)
		for var in ordered_scope:
			self.strides[var] = running_product
			running_product *= self.numPossibleValues[var]

	def _assig(self, index):
		'''
			given the index in the CPD table,
			what is the corresponding assignment
		'''

		assignment = {}
		for i in sorted(self.scope):
			assignment[i] = (index // self.strides[i]) % self.numPossibleValues[i]
		return assignment


	def __str__(self):
		allvars = ( ",".join(map(str, sorted(self.scope))))
		ret = "phi(%s)" %allvars
		ret += "\n%s\n" %('-'*(len(allvars) + 10))
		ret += "|%s | val |" %allvars
		ret += "\n%s\n" %('-'* (len(allvars)  + 10))
		for idx, val in enumerate(self.table):
			currAssignment = ",".join(  map(str, self._assig(idx).values()))
			ret += "%s | %5.4f\n" %( currAssignment, val) 

		ret += "\n%s\n" %("="*50)
		return ret


	def __repr__(self):
		return self.__str__()


	def _idx(self, assignment):
		'''
			given an assignment to each scope value,
			what is the index in the CPD table
		'''
		idx=0
		for var in sorted(self.scope):
			if (var in assignment):
				idx += self.strides[var]*assignment[var]
		return idx


	def normalize(self):
		if (len(self.table) == 0):
			return

		Z = 0.0
		for s in self.table:
			Z += math.exp(s)

		Z = math.log(Z)
		self.table = [val - Z for val in self.table]

	
	def multiply(self, phi2):

		'''
			multiply this factor with phi2
			returning a new factor
		'''

		if (len(phi2.scope) == 0):
			assert(len(phi2.table) == 0)
			return copy.deepcopy(self)

		elif (len(self.scope) == 0):
			assert(len(self.table) == 0)
			return copy.deepcopy(phi2)

		phi_product = Factor()
		phi_product.scope = self.scope | phi2.scope
		phi_product.strides = ddict(int)
		phi_product.numPossibleValues = {}
		phi_product.table = []
		running_product =1

		sorted_scope = sorted(phi_product.scope)

		for var in sorted_scope:
			if (var in self.scope):
				phi_product.numPossibleValues[var] = self.numPossibleValues[var]
			else:
				phi_product.numPossibleValues[var] = phi2.numPossibleValues[var]

			phi_product.strides[var] = running_product
			running_product *= phi_product.numPossibleValues[var]

		# size of the table of the product
		productTableSz = reduce(operator.mul, phi_product.numPossibleValues.values(), 1)
		''' Kohler Algorithm 10.A.1'''
		#print("start multipyling! %d" %productTableSz)

		j = 0
		k = 0
		assignment = {}
		for l in sorted_scope:
			assignment[l]=0 

		for i in xrange(productTableSz):
			phi_product.table.append(self.table[j] + phi2.table[k])
			for l in sorted_scope:
				assignment[l] += 1
				cardinality = phi_product.numPossibleValues[l]
				if assignment[l] == cardinality :
					assignment[l] = 0
					j = j - (cardinality -1)*self.strides[l]
					k = k - (cardinality -1)*phi2.strides[l]

				else:
					j = j + self.strides[l]
					k = k + phi2.strides[l]
					break
		#print("done multiplying!")
		return phi_product

	def divide(self, phi2):
		'''
			divide this factor with phi2
			returning a new factor
		'''
		phi2_aux = copy.deepcopy(phi2)
		phi2_aux.table = [-1*val for val in phi2_aux.table]
		return self.multiply(phi2_aux)

	def marginalize(self, V):
		'''
			-V is a set of nodes over which we want to marginalize over
		'''
		if len(V) == 0:
			return copy.deepcopy(self)

		phi = Factor()
		phi.scope = self.scope - V
		phi.numPossibleValues = {}
		phi.table = []
		phi.strides = ddict(int)

		sorted_scope = sorted(phi.scope)
		running_product =1
		for var in sorted_scope:
			phi.numPossibleValues[var] = self.numPossibleValues[var]
			phi.strides[var] = running_product
			running_product *= phi.numPossibleValues[var]

		varOrdered = sorted(V)

		# size of the new variable
		productTableSz = reduce(operator.mul, phi.numPossibleValues.values(), 1)

		# all assignments to the variables to be marginalized, in order.
		all_assignments = list(itertools.product(*[range(self.numPossibleValues[var]) for var in varOrdered]))
		strideMarginalized = []
		for assignment in all_assignments:
			stride=0
			for i, var in enumerate(varOrdered):
				stride += self.strides[var]*assignment[i] 

			strideMarginalized.append(stride)

		for i in xrange(productTableSz):
			assignment = phi._assig(i)
			base_idx = self._idx(assignment)
			phi.table.append(0)
			for stride in strideMarginalized:
				phi.table[-1] += math.exp(self.table[base_idx + stride])

			phi.table[-1] = math.log(phi.table[-1])

		return phi


	def take_max(self, V):
		'''
			-V is a set of nodes over which we want to take max
		'''
		if len(V) == 0:
			return copy.deepcopy(self)

		phi = Factor()
		phi.scope = self.scope - V
		phi.numPossibleValues = {}
		phi.table = []
		phi.strides = ddict(int)

		sorted_scope = sorted(phi.scope)
		running_product =1
		for var in sorted_scope:
			phi.numPossibleValues[var] = self.numPossibleValues[var]
			phi.strides[var] = running_product
			running_product *= phi.numPossibleValues[var]

		varOrdered = sorted(V)

		# size of the new variable
		productTableSz = reduce(operator.mul, phi.numPossibleValues.values(), 1)

		# all assignments to the variables to be marginalized, in order.
		all_assignments = list(itertools.product(*[range(self.numPossibleValues[var]) for var in varOrdered]))
		strideMarginalized = []
		for assignment in all_assignments:
			stride=0
			for i, var in enumerate(varOrdered):
				stride += self.strides[var]*assignment[i] 

			strideMarginalized.append(stride)

		for i in xrange(productTableSz):
			assignment = phi._assig(i)
			base_idx = self._idx(assignment)
			phi.table.append(0)
			ans = -(1<<20)
			for stride in strideMarginalized:
				ans = max(self.table[base_idx + stride], ans)

			phi.table[-1] = ans

		return phi


class cliqueNode:
	'''
		Nodes of a cliqueTree.
		Each cliqueTree is associated with
		-set of factors
		-scope
		-set of neighbors
		-unique id
		-current belief
	'''

	def __init__(self, _scope = set(), _factors = set()):
		self.scope = _scope
		self.belief = None
		self.factors = _factors
		if len(_factors):
			factors = list(_factors)
			self.factor_product = copy.deepcopy(factors[0])
			for i in xrange(1, len(factors)):
				self.factor_product = (self.factor_product).multiply(factors[i])

		else:
			self.factor_product = Factor()

	def __str__(self):  
		allvars = ( ",".join(map(str, self.scope)))
		ret = "psi(%s)" %allvars
		factor_rep = ["phi(%s)" %",".join(map(str,factor.scope)) for factor in self.factors] 
		ret += "\tfactor product: %s" %(" ".join(factor_rep))
		if len(self.factors):
			ret += "\n factor table size: %d" %(len(self.factor_product.table))
		return ret


	def propogate_message(self, messages):
		delta_message = copy.deepcopy(self.factor_product) 
		for message in messages:
			delta_message = (delta_message).multiply(message)

		self.belief = copy.deepcopy(delta_message)
		return delta_message


