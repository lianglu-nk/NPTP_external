#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
from __future__ import print_function
from __future__ import division
import math
import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import sklearn
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.metrics
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
#import imp
import os
import sys
import tensorflow as tf
import utilsLib
import actLib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
gpu_options=tf.ConfigProto()
gpu_options.gpu_options.allow_growth=True

import time
import gc
import argparse

basePath=os.getcwd()
catalog=basePath.split('python_code')[0]
methodPath = basePath+'/python_code/weave/'

#np.set_printoptions(threshold='nan')
np.set_printoptions(threshold=1000)
np.set_printoptions(linewidth=160)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=15)
np.set_printoptions(suppress=True)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:.2f}'.format



parser = argparse.ArgumentParser()
parser.add_argument("-maxProc", help="Max. Nr. of Processes", type=int, default=3)
parser.add_argument("-availableGPUs", help="Available GPUs", nargs='*', type=int, default=[0])
parser.add_argument("-sizeFact", help="Size Factor GPU Scheduling", type=float, default=1.0)
parser.add_argument("-originalData", help="Path for original data", type=str,default=catalog+'/test_data/')
parser.add_argument("-featureoutname", help="pckl file name", type=str, default="test")
parser.add_argument("-dataset", help="Dataset Name", type=str, default="graphWeave")
parser.add_argument("-saveBasePath", help="saveBasePath", type=str,default=catalog+'/res_test_data/')
parser.add_argument("-ofolds", help="Outer Folds", nargs='+', type=int, default=[0,1,2])
parser.add_argument("-continueComputations", help="continueComputations", action='store_true')
parser.add_argument("-saveComputations", help="saveComputations", action='store_true', default=True)
parser.add_argument("-startMark", help="startMark", type=str, default="start")
parser.add_argument("-finMark", help="finMark", type=str, default="finished")
parser.add_argument("-epochs", help="Nr. Epochs", type=int, default=300)
args = parser.parse_args()



maxProcesses=args.maxProc
availableGPUs=args.availableGPUs
sizeFact=args.sizeFact

dataPathSave=args.originalData
featureoutname = args.featureoutname
datasetName=args.dataset
saveBasePath=args.saveBasePath
if not os.path.exists(saveBasePath):
  os.makedirs(saveBasePath)
savePath=saveBasePath+datasetName+"/"
if not os.path.exists(savePath):
  os.makedirs(savePath)  
dbgPath=savePath+"dbg/"
if not os.path.exists(dbgPath):
  os.makedirs(dbgPath)

compOuterFolds=args.ofolds

continueComputations=args.continueComputations
saveComputations=args.saveComputations
startMark=args.startMark
finMark=args.finMark

nrEpochs=args.epochs
batchSize=128



exec(open(methodPath+'hyperparams.py').read(), globals())



exec(open(methodPath+'loadData.py').read(), globals())

normalizeGlobalDense=False
normalizeGlobalSparse=False
normalizeLocalDense=False
normalizeLocalSparse=False
if not denseInputData is None:
  normalizeLocalDense=True
if not sparseInputData is None:
  normalizeLocalSparse=True
exec(open(methodPath+'prepareDatasetsGlobal.py').read(), globals())



minibatchesPerReportTrain=int(int(np.mean([len(x) for x in folds]))/batchSize)*20
minibatchesPerReportTest=int(int(np.mean([len(x) for x in folds]))/batchSize)



useDenseOutputNetTrain=True
useDenseOutputNetPred=True
computeTrainPredictions=True
compPerformanceTrain=True
computeTestPredictions=True
compPerformanceTest=True



#runningProc=list()



for outerFold in compOuterFolds:
  
  
  
  if len(availableGPUs)>0.5:
    initGPUDeviceAlloc=outerFold%len(availableGPUs)
    usedGPUDeviceAlloc=initGPUDeviceAlloc
  
  
  
  if len(availableGPUs)>0.5:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(availableGPUs[usedGPUDeviceAlloc])
  
  
  
  savePrefix0="o"+'{0:04d}'.format(outerFold+1)
  savePrefix=savePath+savePrefix0
  if os.path.isfile(savePrefix+"."+finMark+".pckl") and (not continueComputations):
    continue
  saveFilename=savePrefix+"."+startMark+".pckl"
  if os.path.isfile(saveFilename):
    continue
  saveFile=open(saveFilename, "wb")
  startNr=0
  pickle.dump(startNr, saveFile)
  saveFile.close()
  dbgOutput=open(dbgPath+savePrefix0+".dbg", "w")
  
  
  
  perfFiles=[]
  takeMinibatch=[]
  for innerFold in range(0, len(folds)):
    if innerFold==outerFold:
      continue
    perfFiles.append([])
    for paramNr in range(0, hyperParams.shape[0]):
      perfFiles[-1].append(savePath+"o"+'{0:04d}'.format(outerFold+1)+"_i"+'{0:04d}'.format(innerFold+1)+"_p"+'{0:04d}'.format(hyperParams.index.values[paramNr])+".test.auc.pckl")
    
    if outerFold<0:
      compNrMinibatches=float(nrEpochs)*math.ceil(float(len(list(set(allSamples)-set(folds[innerFold]))))/float(batchSize))
    else:
      compNrMinibatches=float(nrEpochs)*math.ceil(float(len(list(set(allSamples)-set(folds[innerFold]+folds[outerFold]))))/float(batchSize))
    compLastMinibatch=math.trunc(compNrMinibatches/minibatchesPerReportTest)-1
    takeMinibatch.append(compLastMinibatch)
    
  paramNr, perfTable, perfTableOrig=utilsLib.bestSettingsSimple(perfFiles, hyperParams.shape[0], takeMinibatch)
  print(hyperParams.iloc[paramNr], file=dbgOutput)
  
  
  
  if outerFold<0:
    trainSamples=list(set(allSamples))
    testSamples=list(set(allSamples))
    
    useDenseOutputNetPred=True
    compPerformanceTest=False
  else:
    trainSamples=list(set(allSamples)-set(folds[outerFold]))
    testSamples=folds[outerFold]
    
    useDenseOutputNetPred=True
    compPerformanceTest=True
  exec(open(methodPath+'prepareDatasetsLocal.py').read(), globals())
  
  
  
  exec(open(methodPath+'models.py').read(), globals())
  currentLR=hyperParams.iloc[paramNr].learningRate
  
  
  
  reportTrainAUC=[]
  reportTrainAP=[]
  reportTestAUC=[]
  reportTestAP=[]
  startEpoch=0
  minibatchCounterTrain=0
  minibatchCounterTest=0
  minibatchReportNr=0
  
  
  
  if continueComputations:
    exec(open(methodPath+'step1Load.py').read(), globals())
  endEpoch=nrEpochs
  saveScript=methodPath+'step1Save.py'
  exec(open(methodPath+'runEpochs.py').read(), globals())
  if saveComputations:
    exec(open(methodPath+'step1Save.py').read(), globals())
  
  
  
  exec(open(methodPath+'runPredict.py').read(), globals())
  
  
  
  saveFilename=savePrefix+".finaltest.auc.pckl"
  saveFile=open(saveFilename, "wb")
  pickle.dump(sumTestAUC, saveFile)
  saveFile.close()

  saveFilename=savePrefix+".finaltrain.auc.pckl"
  saveFile=open(saveFilename, "wb")
  pickle.dump(sumTrainAUC, saveFile)
  saveFile.close()
  
  
  
  saveFilename=savePrefix+".evalPredict.hdf5"
  saveFile=h5py.File(saveFilename, "w")
  saveFile.create_dataset('predictions', data=predDenseTest)
  saveFile.close()

  saveFilename=savePrefix+".eval"
  np.savetxt(saveFilename+".cmpNames", np.array(testSamples), fmt="%s")
  np.savetxt(saveFilename+".targetNames", np.array(targetAnnInd.index.values), fmt="%s")
  
  
  
  dbgOutput.close()
