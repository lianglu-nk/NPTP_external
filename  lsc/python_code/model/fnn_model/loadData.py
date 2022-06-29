#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import pickle
from sklearn.feature_selection import VarianceThreshold



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



#denseOutputData=targetMat.A
denseOutputData=None
sparseOutputData=targetMat



if datasetName=="ecfp6":
  f=open(dataPathSave+featureoutname+'ecfp6dense.pckl', "rb")
  ecfpMat=pickle.load(f)
  sampleECFPInd=pickle.load(f)
  # featureStaticInd=pickle.load(f)
  f.close()

  denseInputData = ecfpMat
  denseSampleIndex = sampleECFPInd
  sparseInputData = None
  sparseSampleIndex = None

  del ecfpMat
  del sampleECFPInd
elif datasetName=="fcfp6":
  f=open(dataPathSave+featureoutname+'fcfp6dense.pckl', "rb")
  fcfpMat=pickle.load(f)
  sampleFCFPInd=pickle.load(f)
  # featureSemiInd=pickle.load(f)
  f.close()

  denseInputData = fcfpMat
  denseSampleIndex = sampleFCFPInd
  sparseInputData = None
  sparseSampleIndex = None

  del fcfpMat
  del sampleFCFPInd
elif datasetName=="MACCS":
  f=open(dataPathSave+featureoutname+'MACCSFNNdense.pckl', "rb")
  MACCSMat=pickle.load(f)
  sampleMACCSInd=pickle.load(f)
  # featureECFPInd=pickle.load(f)
  f.close()

  denseInputData = MACCSMat
  denseSampleIndex = sampleMACCSInd
  sparseInputData = None
  sparseSampleIndex = None
  
  del MACCSMat
  del sampleMACCSInd
elif datasetName=="ecfp6fcfp6":
  f=open(dataPathSave+featureoutname+'ecfp6dense.pckl', "rb")
  ecfpMat=pickle.load(f)
  sampleECFPInd=pickle.load(f)
  # featureDFSInd=pickle.load(f)
  f.close()

  f = open(dataPathSave+featureoutname+'fcfp6dense.pckl', "rb")
  fcfpMat = pickle.load(f)
  sampleFCFPInd = pickle.load(f)
  # featureSemiInd=pickle.load(f)
  f.close()

  denseInputData = np.hstack([ecfpMat, fcfpMat])
  denseSampleIndex = sampleECFPInd
  sparseInputData = None
  sparseSampleIndex = None
  
  del ecfpMat
  del sampleECFPInd
  del fcfpMat
  del sampleFCFPInd

elif datasetName=="ecfp6MACCS":
  f=open(dataPathSave+featureoutname+'ecfp6dense.pckl', "rb")
  ecfpMat=pickle.load(f)
  sampleECFPInd=pickle.load(f)
  # featureECFPInd=pickle.load(f)
  f.close()
  
  f=open(dataPathSave+'noweakchembl26_26MACCSFNNdense.pckl', "rb")
  MACCSMat=pickle.load(f)
  sampleMACCSInd=pickle.load(f)
  # featureToxInd=pickle.load(f)
  f.close()

  denseInputData = np.hstack([ecfpMat, MACCSMat])
  denseSampleIndex = sampleECFPInd
  sparseInputData = None
  sparseSampleIndex = None
  
  del ecfpMat
  del sampleECFPInd
  del MACCSMat
  del sampleMACCSInd
  
elif datasetName=="fcfp6MACCS":
  f = open(dataPathSave+featureoutname+'fcfp6dense.pckl', "rb")
  fcfpMat = pickle.load(f)
  sampleFCFPInd = pickle.load(f)
  # featureSemiInd=pickle.load(f)
  f.close()
  
  f=open(dataPathSave+featureoutname+'MACCSFNNdense.pckl', "rb")
  MACCSMat=pickle.load(f)
  sampleMACCSInd=pickle.load(f)
  # featureToxInd=pickle.load(f)
  f.close()

  denseInputData = np.hstack([fcfpMat, MACCSMat])
  denseSampleIndex = sampleFCFPInd
  sparseInputData = None
  sparseSampleIndex = None
  
  del fcfpMat
  del sampleFCFPInd
  del MACCSMat
  del sampleMACCSInd

elif datasetName == "ecfp6fcfp6MACCS":
  f = open(dataPathSave+featureoutname+'ecfp6dense.pckl', "rb")
  ecfpMat = pickle.load(f)
  sampleECFPInd = pickle.load(f)
  # featureECFPInd=pickle.load(f)
  f.close()

  f = open(dataPathSave+featureoutname+'fcfp6dense.pckl', "rb")
  fcfpMat = pickle.load(f)
  sampleFCFPInd = pickle.load(f)
  # featureSemiInd=pickle.load(f)
  f.close()

  f = open(dataPathSave+featureoutname+'MACCSFNNdense.pckl', "rb")
  MACCSMat = pickle.load(f)
  sampleMACCSInd = pickle.load(f)
  # featureToxInd=pickle.load(f)
  f.close()

  denseInputData = np.hstack([ecfpMat, fcfpMat, MACCSMat])
  denseSampleIndex = sampleECFPInd
  sparseInputData = None
  sparseSampleIndex = None

  del ecfpMat
  del sampleECFPInd
  del fcfpMat
  del sampleFCFPInd
  del MACCSMat
  del sampleMACCSInd

gc.collect()



allSamples=np.array([], dtype=np.int64)
if not (denseInputData is None):
  allSamples=np.union1d(allSamples, denseSampleIndex.index.values)
if not (sparseInputData is None):
  allSamples=np.union1d(allSamples, sparseSampleIndex.index.values)
if not (denseInputData is None):
  allSamples=np.intersect1d(allSamples, denseSampleIndex.index.values)
if not (sparseInputData is None):
  allSamples=np.intersect1d(allSamples, sparseSampleIndex.index.values)
allSamples=allSamples.tolist()



if not (denseInputData is None):
  folds=[np.intersect1d(fold, denseSampleIndex.index.values).tolist() for fold in folds]
if not (sparseInputData is None):
  folds=[np.intersect1d(fold, sparseSampleIndex.index.values).tolist() for fold in folds]
