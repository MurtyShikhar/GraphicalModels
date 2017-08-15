#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <unordered_map>

#define mp make_pair
using namespace std;

typedef vector<vector<double> > ocrFactor;
typedef vector<vector<double> > transFactor;

class ocrNet {
private:
	ocrFactor table;
	unordered_map<char,int> charMap;
	string allChars;
public:
	ocrNet(ocrFactor& table): table(table){
		allChars = "doirahtnse";
		for (int i=0; i < allChars.size(); i++)
			charMap[allChars[i]]= i; 
	}
	double partitionFn(vector<int>& img) {
		double denom=0;
		int sz =img.size();
		string allChars = "etaoinshrd";
		for (int i=0; i < sz; i++) {
			double curr=0;
			int imgId = img[i];
			for (int j=0; j < 10; j++) 
				curr += table[imgId][j];
			denom += log(curr);
		}

		return denom;
	}

	double potential(vector<int>& img, string& word){
		double numer=0;
		int sz = img.size();
		for (int i=0; i < sz; i++) {
			int c = charMap[word[i]];
			int imgId = img[i];
			numer += log(table[imgId][c]);
		}

		return numer;
	}

	string inference(vector<int>& img) {
		int sz = img.size();
		string ans="";
		for (int i=0; i < sz; i++) {
			double maxs = -(1<<30);
			int best;
			for (int j=0; j < 10; j++){
				double score=table[img[i]][j];
				if (score > maxs){
					best = j;
					maxs=score;
				} 
			}

			ans.push_back(allChars[best]);
		}
		return ans;
	}
};
