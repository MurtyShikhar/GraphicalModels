#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <unordered_map>
#include <iostream>

#define mp make_pair
using namespace std;

typedef vector<vector<double> > ocrFactor;
typedef vector<vector<double> > transFactor;

class transNet {
private:
	ocrFactor ocrTable;
	transFactor transTable;
	unordered_map<char,int> charMap;
	string allChars;
public:
	transNet(ocrFactor& o, transFactor& t): ocrTable(o), transTable(t){
		allChars = "doirahtnse";
		for (int i=0; i < allChars.size(); i++)
			charMap[allChars[i]]= i; 

	}
	double partitionFn(vector<int>& img) {
		double denom=0;
		int sz = img.size();
		vector<vector<double> > dp(sz, vector<double>(10,0));
		for (int i=0; i < 10; i++){
			int imgId=img[0];
			dp[0][i] = ocrTable[imgId][i];
		}

		for (int i=1; i < sz; i++) {
			int imgId = img[i];
			for (int j=0; j < 10; j++) {
				double sum=0;
				for (int k=0; k < 10; k++)
					sum += dp[i-1][k]*transTable[k][j];
				dp[i][j] = ocrTable[imgId][j]*sum;
			}
		}

		for (int i=0; i < 10; i++)
			denom += dp[sz-1][i];
		return log(denom);
	}

	double potential(vector<int>& img, string& word){
		double numer=0;
		int sz = img.size();
		for (int i=0; i < sz; i++) {
			int c = charMap[word[i]];
			int imgId = img[i];
			numer += log(ocrTable[imgId][c]);
			if (i+1 < sz) {
				int cNext = charMap[word[i+1]];
				numer += log(transTable[c][cNext]);
			}
		}
		return numer;
	}

	string inference(vector<int>& img) {
		// dp[i][t] = max over w0...wi-1 with wi set to t
		int sz = img.size();

		vector<vector<double> > dp(sz, vector<double>(10,0));
		vector<vector<int> > bp(sz, vector<int>(10,0));
		for (int i=0; i < 10; i++){
			int imgId=img[0];
			dp[0][i] = ocrTable[imgId][i];
		}

		for (int i=1; i < sz; i++) {
			int imgId = img[i];
			for (int j=0; j < 10; j++) {
				double best=-(1<<30);
				int choice;
				for (int k=0; k < 10; k++){
					double score = dp[i-1][k]*transTable[k][j];
					if (score > best){
						best=score; choice=k;
					}
				}
				dp[i][j] = ocrTable[imgId][j]*best;
				bp[i][j] = choice;
			}
		}

		int start; double best = -(1<<30);
		for (int i=0; i < 10; i++) {
			double score = dp[sz-1][i];
			if (score > best){
				best = score;
				start=i;
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

		reverse(ans.begin(), ans.end());
		return ans;
	}
};