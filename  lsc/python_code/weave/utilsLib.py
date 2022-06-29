#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
from __future__ import absolute_import, division, print_function
import numbers
import numpy as np
import sklearn
import sklearn.metrics
import pickle
import pandas as pd

def calculateAUCs(t, p):
  aucs = []
  for i in range(p.shape[1]):
    targ = t[:, i] > 0.5
    pred = p[:, i]
    idx = np.abs(t[:, i]) > 0.5
    try:
      aucs.append(sklearn.metrics.roc_auc_score(targ[idx], pred[idx]))
    except ValueError:
      aucs.append(np.nan)
  return aucs
  
def bestSettingsSimple(perfFiles, nrParams, takeMinibatch=[-1,-1,-1]):
  aucFold=[]
  for foldInd in range(0, len(perfFiles)):
    innerFold=-1
    aucParam=[]
    for paramNr in range(0, nrParams):
      #try:
      saveFile=open(perfFiles[foldInd][paramNr], "rb")
      aucRun=pickle.load(saveFile)
      saveFile.close()
      #except:
      #  pass
      if(len(aucRun)>0):
        if takeMinibatch[foldInd]<len(aucRun):
          aucParam.append(aucRun[takeMinibatch[foldInd]])
        else:
          aucParam.append(aucRun[-1])
    
    aucParam=np.array(aucParam)
    
    if(len(aucParam)>0):
      aucFold.append(aucParam)
  aucFold=np.array(aucFold)
  aucMean=np.nanmean(aucFold, axis=0)
  paramInd=np.nanmean(aucMean, axis=1).argmax()
  
  return (paramInd, aucMean, aucFold)
