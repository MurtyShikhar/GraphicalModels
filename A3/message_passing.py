from factor import *
from helpers import *


def dfs_traversal(cliqueTree, clique, parent, i, parents_pointer, children_pointers, levels, marked):
	parents_pointer[clique] = parent
	
	for neighbor in cliqueTree[clique]:
		if (neighbor is not parent):
			marked.add(neighbor)
			children_pointers[clique].add(neighbor)
			dfs_traversal(cliqueTree, neighbor, clique, i+1, parents_pointer, children_pointers, levels, marked)

	levels.append(clique)

def two_way_msg_passing(cliqueTree, useMAP=False):

	deltas = {}

	marked = set()

	for node in cliqueTree:
		if node not in marked:
			marked.add(node)
			parents_pointer  ={node: None}
			children_pointers =ddict(set)
			levels = []
			#print("tree structure")

			dfs_traversal(cliqueTree, node, None, 0, parents_pointer, children_pointers, levels, marked)
			#print("-"*50)

			for node in levels:
				messages = []
				#print "current node:",
				#print(node)
				#print("deltas:")
				#print("neighbors")
				for child in children_pointers[node]:
					messages.append(deltas[(child, node)])
					#print(messages[-1])

				#print("-"*50)
				message = node.propogate_message(messages)

				if parents_pointer[node]:
					parent_node = parents_pointer[node]
					scope_intersecting = (node.scope & parent_node.scope)
					if useMAP:
						deltas[(node, parents_pointer[node])] = message.take_max(node.scope - scope_intersecting)
					else:						
						deltas[(node, parents_pointer[node])] = message.marginalize(node.scope - scope_intersecting)


			for node in levels[::-1]:
				parent = parents_pointer[node]
				if parent:
					node.belief = (node.belief).multiply(deltas[(parent, node)])
				for next_node in children_pointers[node]:
					scope_intersecting = node.scope & next_node.scope
					if useMAP:
						message = (node.belief.divide(deltas[(next_node, node)])).take_max(node.scope - scope_intersecting)					
					else:	
						message = (node.belief.divide(deltas[(next_node, node)])).marginalize(node.scope - scope_intersecting)
					deltas[(node, next_node)] = message


	return cliqueTree


