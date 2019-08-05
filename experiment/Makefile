n=1;
exp=l1cache

.ONESHELL:
all c:
	gcc -std=c99 -lpthread $(exp).c -o experiment;
	time ./experiment $n


.ONESHELL:
all cpp:
	g++ -lpthread indirectbranch.cpp -o experiment;
	time ./experiment $n