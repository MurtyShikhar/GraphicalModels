from factor import *
from helpers import *
import sys
import time

def convergence(curr_factor, prev_factor, eps):
	assert(curr_factor.scope == prev_factor.scope)
	assert(len(curr_factor.table) == len(prev_factor.table))

	for i in xrange(len(curr_factor.table)):
		if abs(curr_factor.table[i] - prev_factor.table[i]) > eps:
			return False

	return True


def create_bethe_cluser_graph(mrf, factors):
	'''
		create a bethe cluser graph from an mrf and a set of factors
	'''

	bethe_graph = {}

	# create a node for each singleton variable
	for nodeId in mrf:
		nodeScope = set([nodeId])
		betheNode = cliqueNode(nodeScope, [])
		bethe_graph[betheNode] = set()

	used_factors = set()
	for factor in factors:
		if len(factor.scope) == 2 and factor not in used_factors:
			used_factors.add(factor)
			node_scope = factor.scope
			node_factors = set([copy.deepcopy(factor)])
			for factor2 in factors:
				if (factor2.scope).issubset(node_scope) and factor2 not in used_factors:
					node_factors.add(copy.deepcopy(factor2))
					used_factors.add(factor2)

			node_curr = cliqueNode(node_scope, node_factors)
			node_neigbors = set()
			for neighbor in bethe_graph:
				if (neighbor.scope).issubset(node_scope):
					node_neigbors.add(neighbor)
					bethe_graph[neighbor].add(node_curr)

			bethe_graph[node_curr] = node_neigbors

	# nodes = bethe_graph.keys()
	# nodes = sorted(nodes, key = lambda x: len(x.scope))
	# for node in nodes:
	# 	print(node)
	# 	print("neighbors")
	# 	for neighbor in bethe_graph[node]:
	# 		print(neighbor)

	# 	print('-'*50)
	return bethe_graph


def loopy_bp(bethe_graph, eps = 1.0e-10, run_for=30.0, useMAP=False):
	# make sure to normalize all factors
	# print(eps)
	# print(run_for)
	deltas_prev = {}
	deltas_curr = {}
	factor_products = {}

	for node in bethe_graph:
		node.propogate_message([])
		if len(node.scope) == 1:
			node.belief = Factor(node.scope, [0.0 for _ in xrange(10)], {var: 10 for var in node.scope})

		(node.belief).normalize() #TODO

		for neighbor in bethe_graph[node]:
			scope = node.scope & neighbor.scope
			table = [0.0 for _ in xrange(10)]
			numPossibleValues = {var: 10 for var in scope}
			deltas_prev[(node, neighbor)] = Factor(scope, table, numPossibleValues)
			deltas_prev[(node, neighbor)].normalize()


	converged = False
	begin = time.time()
	time_elased = 0.0
	while not converged and time_elased < run_for:
		#print("here")
		converged = True
		for node in bethe_graph:
			for neighbor in bethe_graph[node]:
				curr_delta = (node.belief).divide(deltas_prev[(neighbor, node)])
				curr_scope = node.scope & neighbor.scope

				if useMAP:
					deltas_curr[(node, neighbor)] = curr_delta.take_max(node.scope - curr_scope)
				else:
					deltas_curr[(node, neighbor)] = curr_delta.marginalize(node.scope - curr_scope)

				if not convergence(deltas_curr[(node, neighbor)], deltas_prev[(node, neighbor)], eps):
					converged = False 

		for node in bethe_graph:
			messages = []
			for neighbor in bethe_graph[node]:
				messages.append(deltas_curr[neighbor, node])

			belief_prev = copy.deepcopy(node.belief)
			node.propogate_message(messages)
			(node.belief).normalize()
			if not convergence(node.belief, belief_prev, eps):
				converged = False

		for key in deltas_prev:
			deltas_prev[key] = copy.deepcopy(deltas_curr[key])

		time_elased = time.time() - begin
		if converged:
			print("converged.")

	return bethe_graph




