#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import pickle
import rdkit
import rdkit.Chem
import deepchem
import deepchem.feat
import pickle



f=open(dataPathSave+'folds0.pckl', "rb")
folds=pickle.load(f)
f.close()

f=open(dataPathSave+'labelsHard.pckl', "rb")
targetMat=pickle.load(f)
sampleAnnInd=pickle.load(f)
targetAnnInd=pickle.load(f)
f.close()

targetMat=targetMat
targetMat=targetMat.copy().tocsr()
targetMat.sort_indices()
targetAnnInd=targetAnnInd
targetAnnInd=targetAnnInd-targetAnnInd.min()

folds=[np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
targetMatTransposed=targetMat[sampleAnnInd[list(itertools.chain(*folds))]].T.tocsr()
targetMatTransposed.sort_indices()
trainPosOverall=np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
trainNegOverall=np.array([np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])



denseOutputData=targetMat.A
sparseOutputData=None
#denseOutputData=None
#sparseOutputData=targetMat



if datasetName=="lstm_ecfp6fcfp6MACCS":
  f=open(dataPathSave+featureoutname+'LSTM.pckl', "rb")
  rdkitArr=pickle.load(f)
  f.close()

  f=open(dataPathSave+featureoutname+"Smiles.pckl",'rb')
  smilesArr=pickle.load(f)
  f.close()
  
  denseInputData=None
  denseSampleIndex=None
  sparseInputData=None
  sparseSampleIndex=None
  lstmGraphInputData=rdkitArr
  lstmSmilesInputData=smilesArr
  lstmSampleIndex=sampleAnnInd
  
  f=open(dataPathSave+featureoutname+"MACCS.pckl",'rb')
  maccsMat=pickle.load(f)
  f.close()
  
  f=open(dataPathSave+featureoutname+'ecfp6sparse.pckl', "rb")
  ecfpMat=pickle.load(f)
  ecfpMat.data[:]=1.0
  sampleECFPInd=pickle.load(f)
  f.close()

  f=open(dataPathSave+featureoutname+'fcfp6sparse.pckl', "rb")
  fcfpMat=pickle.load(f)
  sampleFCFPInd=pickle.load(f)
  f.close()
  
  maccsInputData=maccsMat
  ecfpInputData=ecfpMat
  fcfpInputData=fcfpMat

gc.collect()



allSamples=np.array([], dtype=np.int64)
if not (lstmSmilesInputData is None):
  allSamples=np.union1d(allSamples, lstmSampleIndex.index.values)
if not (lstmSmilesInputData is None):
  allSamples=np.intersect1d(allSamples, lstmSampleIndex.index.values)
allSamples=allSamples.tolist()



if not (lstmSampleIndex is None):
  folds=[np.intersect1d(fold, lstmSampleIndex.index.values).tolist() for fold in folds]
