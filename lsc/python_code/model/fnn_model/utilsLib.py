#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
from __future__ import absolute_import, division, print_function
import numbers
import numpy as np
import sklearn
import sklearn.metrics
import pickle
import pandas as pd


def calculateSparseAUCs(t, p):
    aucs = []
    for i in range(p.shape[0]):
        targ = t[i].data > 0.5
        pred = p[i].data
        try:
            aucs.append(sklearn.metrics.roc_auc_score(targ, pred))
        except ValueError:
            aucs.append(np.nan)
    return aucs

    
def bestSettingsSimple(perfFiles, nrParams, takeMinibatch=[[-1, -1], [-1, -1], [-1, -1], [-1,-1]]):
    aucFold=[]
    for outind in range(0,3):
        for foldInd in range(0, 2):
            aucParam=[]
            for paramNr in range(0, nrParams):
                #try:
                saveFile=open(perfFiles[outind][foldInd][paramNr], "rb")
                aucRun=pickle.load(saveFile)
                saveFile.close()
                #except:
                #  pass
                if (len(aucRun) > 0):
                    if takeMinibatch[outind][foldInd] < len(aucRun):
                        aucParam.append(aucRun[takeMinibatch[outind][foldInd]])
                    else:
                        aucParam.append(aucRun[-1])
        
            aucParam=np.array(aucParam)
        
            if(len(aucParam)>0):
                aucFold.append(aucParam)
    aucFold=np.array(aucFold)
    aucMean=np.nanmean(aucFold, axis=0)
    paramInd=np.nanmean(aucMean, axis=1).argmax()
  
    return (paramInd, aucMean, aucFold)
