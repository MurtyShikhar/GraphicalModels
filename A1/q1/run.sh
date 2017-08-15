bayesDir="bayesOut.txt"
queryDir="query.txt"
if [ $1 -eq 0 ]; then
	./bayesNet.o generateNet $bayesDir $2 $3
	python bayes/display.py $bayesDir
else
	./bayesNet.o dseparation queryOut.txt $bayesDir $queryDir
	python bayes/display.py $bayesDir queryOut.txt
fi
