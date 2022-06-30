#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr

ifeq ($(multiproc), 1)
	MP = -fopenmp -Dmultiproc
endif
ifeq ($(debug), 1)
	DBG = -g
endif
ifeq ($(debugMat), 1)
	DBG = -g -Ddbgmat
endif
ifeq ($(optim), 1)
	OPT = -O2
endif

CF = -lgsl -lgslcblas -lm $(MP) $(DBG)

all: genDirStructure convertFPFBinary clusterMinFull

genDirStructure:
	g++ genDirStructure.cpp -o genDirStructure.exec $(CF)
convertFPFBinary:
	g++ convertFPFBinary.cpp -o convertFPFBinary.exec $(CF)
clusterMinFull:
	g++ clusterMinFull.cpp -o clusterMinFull.exec -std=c++11 $(MP)
