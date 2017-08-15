#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <vector>
#include <map>
#include <unordered_set>
#include <set>
#include <stack>
#include <chrono>
#include <queue>
#include <cstring>
//#define _DEBUG_
#define mp make_pair
using namespace std;

typedef vector<int> vi;
typedef vector<pair<vi,vi > > network; // a bayesian network is just a vector of pairs containing children vectors and parent vectors
typedef pair<int,int> pii;

// add child to child vector of parent, and parent to parent vector of child
void create_link(int child, int parent, network& net){
	net[child].first.push_back(parent);
	net[parent].second.push_back(child);
}

void generateNet(int n, int k, network& net) {
	net.resize(n);
	// assume topological ordering over nodes.
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dis(0, k);
	vector<int> indices;
	for (int i=0; i < n; i++) {
		// random number between 0 and k
		int children = dis(gen);

		// choose from i+1 to n-1 == (n-1-i), so push remaning indices into "indices"
		int remaining = (n-1-i);
		indices.clear();
		for (int j = i+1; j < n; j++)
			indices.push_back(j);

		// if random number greater than or equal to remaninng children push em all
		if (remaining <= children){
			for (int j=0; j < indices.size(); j++)
				create_link(indices[j], i, net);
		}
		// else randomly take "children" indices.
		else
		{			
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
			for (int j=0; j < children; j++)
				create_link(indices[j], i, net);
		}

	}
}


void readNet(string& inpFile, network& bayesNet) {
	int n;
	ifstream file(inpFile);
	file >> n;
	bayesNet.resize(n);
  	string s;
  	getline(file, s);
  	stringstream ss;
	for (int i=0; i < n; i++) {
		s.clear();
		getline(file,s);
		ss.clear();
		int spaceDelim = s.find(" ");
		s = s.substr(spaceDelim+2, s.size()-(spaceDelim+3));
		ss << s;
		while (getline(ss, s, ',')) {
			int child = (stoi(s)-1);
			create_link(child,i, bayesNet);
		}
	}

	file.close();
}

void writeNet(string& outFile, network& bayesNet) {
	ofstream file(outFile);
	int n = bayesNet.size();
	file << n << "\n";
	string s;
	for (int i=0; i < n; i++) {
		file << i+1 << " [";
		int en = bayesNet[i].second.size();
		for (int j=0; j < en-1; j++) {
			file << (bayesNet[i].second)[j]+1 << ",";
		}

		if (en !=0)
			file << (bayesNet[i].second)[en-1]+1 << "]\n" ;
		else
			file << "]\n";
	}

	file.close();
}

// insert (children,1) of f.first into the bfs queue 
void insert_child(network& bayesNet, pii f, queue<pair<pii,pii> >& bfs) {
	int node = f.first;
	int numChildren = (bayesNet[node].second).size();
	for (int i=0; i < numChildren; i++) {
		int child = (bayesNet[node].second)[i];
		pii exp = make_pair(child,1);
		bfs.push(mp(f,exp));
	}	
}

// insert (parent,0) of f.first into the bfs queue 
void insert_parent(network& bayesNet, pii f, queue<pair<pii,pii> >& bfs) {
	int node = f.first;
	int numParents = (bayesNet[node].first).size();
	for (int i=0; i < numParents; i++) {
		int parent = (bayesNet[node].first)[i];
		pii exp = make_pair(parent,0);
		bfs.push(mp(f,exp));
	}	
}


void dseparation(int xi, int xj, set<int>& z, network& bayesNet, set<pii>& reachable, map<pii,pii>& bp) {
	// do a bfs to mark all given and ancestors of given nodes
	set<int> valid_v;
	queue<int> fifo; auto it = z.begin(); 
	while (it != z.end()) {
		fifo.push(*it);
		it++;
	}

	while (!fifo.empty()) {
		int node = fifo.front(); fifo.pop();
		auto _it = valid_v.find(node);
		if (_it == valid_v.end()) {
			int numParents = (bayesNet[node].first).size();
			for (int i=0; i < numParents; i++) {
				int parent = (bayesNet[node].first)[i];
				fifo.push(parent);
			}
		}
		valid_v.insert(node);
	}

	// bfs queue stores both the parent of the node and the node.
	queue<pair<pii, pii> > bfs;
	bfs.push(mp(mp(xi,0),mp(xi,0)));
	set<pii> visited;
	while (!bfs.empty()) {
		pair<pii,pii> _f = bfs.front(); bfs.pop();
		pii f = _f.second;
		auto it2 = visited.find(f);
		// if node hasn't been visited before only then visit it.
		if (it2 == visited.end()){
			int node = f.first;
			int direction = f.second;
			// mark as visited.
			visited.insert(f);
			reachable.insert(f);
			// set backpointer of f to be parent of f since this was the first parent to visit it.
			bp[f] = _f.first;
			auto it3 = z.find(node);
			// we came to this node from its child 
			if (!direction) {				
			 	if (it3 == z.end()) {
					insert_child(bayesNet, f, bfs);
					insert_parent(bayesNet, f, bfs);
				}
			}

			// we have come to node from its parent 
			else {
				// descend to child of 
				if (it3 == z.end()) 
					insert_child(bayesNet, f, bfs);
				// descend to parent
				auto it4 = valid_v.find(node);
				// valid v structure!
				if (it4 != valid_v.end()) 
					insert_parent(bayesNet, f, bfs);
			}
		}
	}

}

void read_query(network& bayesNet, string& queryFile, string& outputFile) {
	string s;
	ifstream file(queryFile);
	ofstream ofile(outputFile);
	stringstream ss;
	int n;
	file >> n;
	getline(file, s);
	int xi,xj;
  	for (int i=0; i < n; i++){
  		s.clear();
  		getline(file, s);
  		ss.clear();
  		ss << s;
  		ofile << s << '\n';
  		getline(ss, s, ' ');
  		xi = stoi(s)-1;
  		getline(ss, s, ' ');
		xj = stoi(s)-1; 
		getline(ss, s, ' ');
		s = s.substr(1, s.size()-2);	
		ss.clear();
  		ss << s;
  		set<int> z;
  		while (getline(ss, s, ',')) {
  			int node = (stoi(s)-1);
  			z.insert(node);  			
  		}
  		
  		set<pii> reachable;
  		map<pii, pii> bp;
  		dseparation(xi, xj, z, bayesNet, reachable, bp);
  		auto it1 = reachable.find(make_pair(xj,0));
  		auto it2 = reachable.find(make_pair(xj,1));
  		int st = -1;
  		if (it1 != reachable.end()) 
  			st = 0;

  		else if (it2 != reachable.end()) 
  			st = 1;
  		else 
  			ofile << "yes\n";
  			
  		if (st != -1) {
  			ofile << "no ";
  			stack<int> path;
  			pii curr = make_pair(xj,st);
  			pii start = make_pair(xi, 0);
  			while (curr != start) {
  				path.push(curr.first);
  				curr = bp[curr];
  			}

  			path.push(xi);
  			ofile <<"[";
  			while (!path.empty()){
  				int _top = path.top();
  				path.pop();
  				if (path.empty())
  					ofile << _top+1 << "]\n";
  				else
  					ofile << _top+1 << ",";
  			}
  		}
  	}

  	ofile.close();
  	
}

int main(int argc, char* argv[]) {
	if (argc < 5) {
		printf("Usage: arg1: dseparation, arg2: output File arg3<optional>: bayesNet, arg4: query File\n");
		printf("OR\n");
		printf("Usage: arg1: generateNet, arg2: output File arg3: n, arg4: k\n");
		return -1;
	}

	if (!strcmp(argv[1], "generateNet")){
		vector<pair<vi,vi> > net;
		int n = atoi(argv[3]), k = atoi(argv[4]);
		generateNet(n, k, net);
		string outputFile(argv[2]);
		writeNet(outputFile, net);

	}

	else if (!strcmp(argv[1], "dseparation") && argc > 4) {
		vector<pair<vi,vi> > net;
		string outFile(argv[2]);
		string bayesNetFile(argv[3]);
		readNet(bayesNetFile, net);
		string queryFile(argv[4]);
		read_query(net, queryFile, outFile);
	}

	else {
		printf("unknown task.\n");
		return -1;
	}

	
}