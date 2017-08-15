#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#define mp make_pair
using namespace std;

typedef vector<vector<double> > ocrFactor;
typedef vector<vector<double> > transFactor;

class combinedNet {
private:
	ocrFactor ocrTable;
	transFactor transTable;
	ocrFactor ocrTableNoLog;
	transFactor transTableNoLog;
	unordered_map<char,int> charMap;
	string allChars;
	double partitionFnPartial(vector<int>& img, string& partial, vector<vector<double> >& dp, int sz) {
		double denom=0;
		for (int i=0; i < 10; i++){
			int imgId=img[0];
			dp[0][i] = ocrTableNoLog[imgId][i];
		}

		for (int i=1; i < sz; i++) {
			int imgId = img[i];
			for (int j=0; j < 10; j++) {
				double sum=0;
				if (partial[i-1] != '0'){
					int charAtPartialPos = charMap[partial[i-1]];
					sum = dp[i-1][charAtPartialPos]*transTableNoLog[charAtPartialPos][j];
				}
				else{
					for (int k=0; k < 10; k++)
						sum += dp[i-1][k]*transTableNoLog[k][j];
				}

				dp[i][j] = ocrTableNoLog[imgId][j]*sum;
			}
		}

		if (partial[sz-1] != '0'){
			int charAtPartialPos = charMap[partial[sz-1]];
			denom = dp[sz-1][charAtPartialPos];
		}

		else{
			for (int i=0; i < 10; i++)
				denom += dp[sz-1][i];
		}

		return denom;
	}


	pair<double,string> inferencePartial(vector<int>& img, string& partial, vector<vector<double> >& dp, vector<vector<int> >& bp, int sz) {
		// dp[i][t] = max over w0...wi-1 with wi set to t
		for (int i=0; i < 10; i++){
			int imgId=img[0];
			dp[0][i] = ocrTableNoLog[imgId][i];
		}

		for (int i=1; i < sz; i++) {
			int imgId = img[i];
			for (int j=0; j < 10; j++) {
				double best=-(1<<30);
				int choice;
				if (partial[i-1] != '0'){
					choice = charMap[partial[i-1]];
					best = dp[i-1][choice]*transTableNoLog[choice][j];
				}
				else{
					for (int k=0; k < 10; k++){
						double score = dp[i-1][k]*transTableNoLog[k][j];
						if (score > best){
							best=score; choice=k;
						}
					}
				}
				dp[i][j] = ocrTableNoLog[imgId][j]*best;
				bp[i][j] = choice;
			}
		}

		int start; double best = -(1<<30);
		if (partial[sz-1] != '0'){
			int charAtPartialPos = charMap[partial[sz-1]];
			best = dp[sz-1][charAtPartialPos];
			start = charAtPartialPos;
		}
		else{
			for (int i=0; i < 10; i++) {
				double score = dp[sz-1][i];
				if (score > best){
					best = score;
					start=i;
				}
			}
		}

		string ans; ans.push_back(allChars[start]);
		int idx = sz-1;
		int id = start;
		while (idx >=1) {
			id = bp[idx][id];
			ans.push_back(allChars[id]);
			idx--;
		}

		return make_pair(best,ans);
	}


	void createPartials(vector<string>& partials, int idx, int sz, unordered_set<int>& equalIdxs,  string& currWord) {
		if (idx == sz)
			partials.push_back(currWord);
		else {
			auto it = equalIdxs.find(idx);

			if (it != equalIdxs.end()) {
				for (int i=0; i < 10; i++) {
					currWord.push_back(allChars[i]);
					createPartials(partials, idx+1, sz, equalIdxs, currWord);
					currWord.pop_back();
				}
			}

			else {
				currWord.push_back('0');
				createPartials(partials, idx+1, sz, equalIdxs, currWord);
				currWord.pop_back();
			}
		}
	}


public:
	combinedNet(ocrFactor& o, transFactor& t){
		ocrTable = ocrTableNoLog = o;
		transTable=  transTableNoLog =t;
		allChars = "doirahtnse";
		for (int i=0; i < 1000; i++){
			for (int j=0; j < 10; j++){
				ocrTable[i][j] = log(ocrTable[i][j]);
			}
		}
		
		for (int i=0; i < 10; i++){
			for (int j=0; j < 10; j++)
				transTable[i][j] = log(transTable[i][j]);
		}

		for (int i=0; i < allChars.size(); i++)
			charMap[allChars[i]]= i; 
	}

	double potential(vector<int>& img, string& word){
		double numer=0;
		int sz = img.size();
		for (int i=0; i < sz; i++) {
			int c = charMap[word[i]];
			int imgId = img[i];
			numer += ocrTable[imgId][c];
			if (i+1 < sz) {
				int cNext = charMap[word[i+1]];
				numer += transTable[c][cNext];
			}
		}

		double log5 = log(5.0);
		for (int i=0; i < sz; i++) {
			for (int j=i+1; j < sz; j++)
			if (word[i] == word[j])
				numer += log5;
		}

		return numer;
	}
	
	string inference(vector<int>& img, double& partition) {
		int sz = img.size();
		vector<vector<double> > dp(sz, vector<double>(10,0));
		vector<vector<int> > bp(sz, vector<int>(10,0));
		vector<pair<int,int> > equalIdxs;
		unordered_set<int> equals;

		for (int i=0; i < sz; i++){
			for (int j=i+1; j < sz; j++){
				if (img[i]==img[j]){
					equalIdxs.push_back(mp(i,j));
					equals.insert(i); equals.insert(j);
				}
			}
		}

		vector<string> partials;
		string currWord = "";
		createPartials(partials, 0, sz, equals, currWord);

		int sz2 = partials.size();
		int equalIds = equalIdxs.size();
		double log5 = log(5.0);
		double best = -(1<<30);
		string bestAns;
 		for (int i=0; i < sz2; i++) {
 			pair<double,string> ans =  inferencePartial(img, partials[i], dp, bp, sz);
 			double currPart=log(partitionFnPartial(img, partials[i], dp, sz));
 			double curr = ans.first;
 			for (int j=0; j< equalIds; j++){
 				int i1 = equalIdxs[j].first, i2 = equalIdxs[j].second;
 				if (partials[i][i1]==partials[i][i2]){
 					currPart += log5;
 					curr += log5;
 				}
 			}
 			partition += exp(currPart);

			if (curr > best){
				best=curr;
				bestAns=(ans.second);
			}
		}

		reverse(bestAns.begin(), bestAns.end());
		return bestAns;
	}
};
