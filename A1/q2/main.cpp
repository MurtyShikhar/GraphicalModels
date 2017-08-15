#include <vector>
#include <set>
#include <fstream>
#include <cstdio>
#include <map>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <cstring>

#include "transModel.cpp"
#include "ocrModel.cpp"
#include "combinedModel.cpp"

using namespace std;



void read_ocrData(string s, map<char,int>& charMap, ocrFactor& table){
	FILE *fp;
	if ( (fp = fopen(s.c_str(), "r")) ==NULL)
		printf("Cannot open file\n");
	else{

		int img;
		char c;
		double prob;
		while (fscanf(fp, "%d\t%c\t%lf\n", &img, &c, &prob) == 3)
			table[img][charMap[c]] = prob;
		fclose(fp);
	}

} 


void read_transData(string s, map<char,int>& charMap, transFactor& table){
	FILE *fp;
	if ( (fp = fopen(s.c_str(), "r")) ==NULL)
		printf("Cannot open file\n");
	else{
		string allChars = "doirahtnse";

		for (int i=0; i < allChars.size(); i++)
			charMap[allChars[i]]= i; 

		char ci;
		char cnext;
		double prob;
		while (fscanf(fp, "%c\t%c\t%lf\n", &ci, &cnext, &prob) == 3){
			int curr = charMap[ci];
			int nxt  = charMap[cnext];
			table[curr][nxt] = prob;
			//table[curr][nxt] = prob;

		}
		fclose(fp);
	}

} 

template <typename mrf>
void read_imgData(string s, mrf& net, vector<string>& predictions) {
	ifstream file(s);
	string l;
	stringstream ss;
	int i=0;
	while (getline(file, l)) {
		ss.clear(); ss << l;
		i+=1;
		vector<int> img;
		int id;
		while (ss >> id) img.push_back(id);
		string w = net.inference(img);
		predictions.push_back(w);
		cout << w << "\n";
	}

}


void read_CombinedImgData(string s, combinedNet& net, vector<string>& predictions, vector<double>& partitionFn) {
	ifstream file(s);
	string l;
	stringstream ss;
	int i=0;
	while (getline(file, l)) {
		ss.clear(); ss << l;
		i+=1;
		vector<int> img;
		int id;
		while (ss >> id) img.push_back(id);
		double z=0.0;
		string w = net.inference(img, z);
		predictions.push_back(w);
		partitionFn.push_back(z);
		cout <<  w << "\n";
	}

}

void calculateCombinedLikelihood(string imgFile, string wordFile, combinedNet& net, vector<double>& partitionFn) {
	ifstream file1(imgFile);
	ifstream file2(wordFile);
	string l1;
	string l2;
	stringstream ss;
	int i=0;
	double likelihood=0.0;
	while (getline(file1, l1) && getline(file2, l2)) {
		ss.clear(); ss << l1;
		vector<int> img;
		int id;
		while (ss >> id) img.push_back(id);
		double numer = net.potential(img, l2);
		double denom = log(partitionFn[i]);
		likelihood += numer-denom;
		i+=1;
	}

	cout << "log likelihood of data: " << likelihood/i << '\n';
}


template <typename mrf>
void calculateLikelihood(string imgFile, string wordFile, mrf& net) {
	ifstream file1(imgFile);
	ifstream file2(wordFile);
	string l1;
	string l2;
	stringstream ss;
	int i=0;
	double likelihood=0.0;
	while (getline(file1, l1) && getline(file2, l2)) {
		ss.clear(); ss << l1;
		vector<int> img;
		int id;
		while (ss >> id) img.push_back(id);
		double numer = net.potential(img, l2);
		double denom = net.partitionFn(img);
		likelihood += numer-denom;
		i+=1;
	}

	cout << "log likelihood of data: " << likelihood/i << '\n';
}


double charWiseAccuracy(string& gold, string& pred, double& correctChars, double& totalChars) {
	int sz = gold.size();
	double correct=0;
	for (int i=0; i < sz; i++) {
		totalChars++;
		correctChars += (gold[i]==pred[i]);
	}
}

void results(vector<string>& predictions, string goldLabels) {
	ifstream file1(goldLabels);
	string l1;
	int i=0;
	double totalChars=0.0, correctChars=0.0;
	double wordAcc=0.0;
	while (getline(file1, l1)) {
		charWiseAccuracy(l1, predictions[i], correctChars,totalChars);
		if (l1==predictions[i]) wordAcc+=1;
		i+=1;
	}

	cout << "char accuracy: " << correctChars/totalChars << "\n";
	cout << "word accuracy: " << wordAcc/i << "\n";

}



int main(int argc, char* argv[]){

	if (argc < 4) {
		printf("please specify the model, the imgData and the goldLabels.\n");
		printf("*********************************************************\n");
		printf("Usage: arg1: model Name, arg2: imgData, arg3: goldLabels\n");
		return -1;
	}

	else {

		string allChars = "doirahtnse";
		map<char,int> charMap;
		for (int i=0; i < allChars.size(); i++)
			charMap[allChars[i]]= i; 

		ocrFactor ocrTable(1000, vector<double>(10,0.0));
		transFactor transTable(10, vector<double>(10,0.0));
		read_ocrData("OCRdataset/potentials/ocr.dat",charMap, ocrTable);
		read_transData("OCRdataset/potentials/trans.dat",charMap, transTable);

		string imgData(argv[2]);
		string goldLabels(argv[3]);
		vector<string> predictions;
		printf("running %s\n", argv[1]);
		if (!strcmp(argv[1],"ocrNet"))  {
			ocrNet onet(ocrTable);
			read_imgData<ocrNet>(imgData, onet, predictions);
			results(predictions, goldLabels);
			calculateLikelihood<ocrNet>(imgData, goldLabels, onet);
		}

		else if (!strcmp(argv[1],"transNet")) {
			transNet tnet(ocrTable, transTable);
			read_imgData<transNet>(imgData, tnet, predictions);
			results(predictions, goldLabels);
			calculateLikelihood<transNet>(imgData, goldLabels, tnet);
		}

		else if (!strcmp(argv[1],"combinedNet")) {			
			combinedNet cnet(ocrTable, transTable);
			vector<double> partitionFn;
			read_CombinedImgData(imgData, cnet, predictions,partitionFn);
			results(predictions, goldLabels);
			calculateCombinedLikelihood(imgData, goldLabels, cnet, partitionFn);
		}

		else {
			printf("Not a valid model.\n");
			return -1;
		}
	}

}