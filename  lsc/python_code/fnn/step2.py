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

os.environ['CUDA_VISIBLE_DEVICES'] = ''
gpu_options = tf.GPUOptions(allow_growth=True)
import time
import gc
import argparse
import utilsLib
import actLib

basePath=os.getcwd()
catalog=basePath.split('python_code')[0]
methodPath = basePath+'/python_code/fnn/'

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
#parser.add_argument("-datasetNames", help="Dataset Name", type=str, default=["ecfp6", "fcfp6", "MACCS", "ecfp6fcfp6", "ecfp6MACCS", "fcfp6MACCS", "ecfp6fcfp6MACCS"])
parser.add_argument("-datasetNames", help="Dataset Name", type=str, default=["ecfp6fcfp6MACCS"])
parser.add_argument("-saveBasePath", help="saveBasePath", type=str,default=catalog+'/res_test_data/')
parser.add_argument("-ofolds", help="Outer Folds", nargs='+', type=int, default=[0, 1, 2])
parser.add_argument("-continueComputations", help="continueComputations", action='store_true')
parser.add_argument("-saveComputations", help="saveComputations", action='store_true', default=True)
parser.add_argument("-startMark", help="startMark", type=str, default="start")
parser.add_argument("-finMark", help="finMark", type=str, default="finished")
parser.add_argument("-epochs", help="Nr. Epochs", type=int, default=300)
args = parser.parse_args()

maxProcesses = args.maxProc
availableGPUs = args.availableGPUs
sizeFact = args.sizeFact

dataPathSave = args.originalData
featureoutname = args.featureoutname
datasetNames = args.datasetNames
for datasetName in datasetNames:
  saveBasePath = args.saveBasePath
  if not os.path.exists(saveBasePath):
      os.makedirs(saveBasePath)
  savePath = saveBasePath + datasetName + "/"
  if not os.path.exists(savePath):
      os.makedirs(savePath)
  dbgPath = savePath + "dbg/"
  if not os.path.exists(dbgPath):
      os.makedirs(dbgPath)
  
  compOuterFolds = args.ofolds
  
  continueComputations = args.continueComputations
  saveComputations = args.saveComputations
  startMark = args.startMark
  finMark = args.finMark
  
  nrEpochs = args.epochs
  batchSize = 128
  
  exec(open(methodPath + 'hyperparams.py').read(), globals())
  
  exec(open(methodPath + 'loadData.py').read(), globals())
  
  normalizeGlobalDense = False
  normalizeGlobalSparse = False
  normalizeLocalDense = False
  normalizeLocalSparse = False
  if not denseInputData is None:
      normalizeLocalDense = True
  if not sparseInputData is None:
      normalizeLocalSparse = True
  exec(open(methodPath + 'prepareDatasetsGlobal.py').read(), globals())
  
  minibatchesPerReportTrain = int(int(np.mean([len(x) for x in folds])) / batchSize)* 20
  minibatchesPerReportTest = int(int(np.mean([len(x) for x in folds])) / batchSize)
  
  useDenseOutputNetTrain = False
  useDenseOutputNetPred = False
  computeTrainPredictions = True
  compPerformanceTrain = True
  computeTestPredictions = True
  compPerformanceTest = True
  
  runningProc = list()
  
  for outerFold in compOuterFolds:
  
      if len(availableGPUs) > 0.5:
          initGPUDeviceAlloc = outerFold % len(availableGPUs)
          usedGPUDeviceAlloc = initGPUDeviceAlloc
  
      if len(availableGPUs) > 0.5:
          os.environ['CUDA_VISIBLE_DEVICES'] = str(availableGPUs[usedGPUDeviceAlloc])
  
      savePrefix0 = "o" + '{0:04d}'.format(outerFold + 1)
      savePrefix = savePath + savePrefix0
      if os.path.isfile(savePrefix + "." + finMark + ".pckl") and (not continueComputations):
          continue
      saveFilename = savePrefix + "." + startMark + ".pckl"
      saveFile = open(saveFilename, "wb")
      startNr = 0
      pickle.dump(startNr, saveFile)
      saveFile.close()
      dbgOutput = open(dbgPath + savePrefix0 + ".dbg", "w")
      perfFiles = []
      takeMinibatch = []
      for innerFold in range(0, len(folds)):
          if innerFold == outerFold:
              continue
          perfFiles.append([])
          for paramNr in range(0, hyperParams.shape[0]):
              perfFiles[-1].append(savePath + "o" + '{0:04d}'.format(outerFold + 1) + "_i" + '{0:04d}'.format(
                  innerFold + 1) + "_p" + '{0:04d}'.format(hyperParams.index.values[paramNr]) + ".test.auc.pckl")
  
          if outerFold < 0:
              compNrMinibatches = float(nrEpochs) * math.ceil(
                  float(len(list(set(allSamples) - set(folds[innerFold])))) / float(batchSize))
          else:
              compNrMinibatches = float(nrEpochs) * math.ceil(
                  float(len(list(set(allSamples) - set(folds[innerFold] + folds[outerFold])))) / float(batchSize))
          compLastMinibatch = math.trunc(compNrMinibatches / minibatchesPerReportTest) - 1
          takeMinibatch.append(compLastMinibatch)
  
      paramNr, perfTable, perfTableOrig = utilsLib.bestSettingsSimple(perfFiles, hyperParams.shape[0], takeMinibatch)
      print(hyperParams.iloc[paramNr], file=dbgOutput)
  
      if outerFold < 0:
          trainSamples = list(set(allSamples))
          testSamples = list(set(allSamples))
  
          useDenseOutputNetPred = True
          compPerformanceTest = False
          logPerformanceAtBestIter = False
          savePredictionsAtBestIter = False
      else:
          trainSamples = list(set(allSamples) - set(folds[outerFold]))
          testSamples = folds[outerFold]
  
          useDenseOutputNetPred = False
          compPerformanceTest = True
          logPerformanceAtBestIter = False
          savePredictionsAtBestIter = False
      exec(open(methodPath + 'prepareDatasetsLocal.py').read(), globals())
  
      basicArchitecture = hyperParams.iloc[paramNr].basicArchitecture
      if basicArchitecture == "selu":
          exec(open(methodPath + 'modelSELU.py').read(), globals())
      elif basicArchitecture == "relu":
          exec(open(methodPath + 'modelReLU.py').read(), globals())
      currentLR = hyperParams.iloc[paramNr].learningRate
      currentDropout = hyperParams.iloc[paramNr].dropout
      currentIDropout = hyperParams.iloc[paramNr].idropout
      currentL1Penalty = hyperParams.iloc[paramNr].l1Penalty
      currentL2Penalty = hyperParams.iloc[paramNr].l2Penalty
      currentMom = hyperParams.iloc[paramNr].mom
  
      session.run(init)
      session.run(biasInitOp, feed_dict={biasInit: trainBias.astype(np.float32)})
      if (normalizeGlobalSparse or normalizeLocalSparse) and (nrSparseFeatures > 0.5):
          session.run(sparseMeanInitOp, feed_dict={sparseMeanInit: trainSparseDiv2.reshape(1, -1)})
          session.run(sparseMeanWSparseOp.op)
  
      if basicArchitecture == "selu":
          session.run(scaleTrainId, feed_dict={inputDropout: currentIDropout, hiddenDropout: currentDropout})
          session.run(scaleTrainHd, feed_dict={inputDropout: currentIDropout, hiddenDropout: currentDropout})
      elif basicArchitecture == "relu":
          myweightTensors = weightTensors.copy()
          myweightTensors[1] = myweightTensors[1][0]
  
          np.random.seed(123)
          for tenNr in range(1, len(myweightTensors)):
              n_inputs = int(myweightTensors[tenNr].get_shape()[0])
              n_outputs = int(myweightTensors[tenNr].get_shape()[1])
  
              s = np.sqrt(6) / np.sqrt(n_inputs)
              initTen = np.random.uniform(-s, +s, (n_outputs, n_inputs)).T
              session.run(myweightTensors[tenNr].assign(initTen).op)
  
      reportTrainAUC = []
      reportTrainAP = []
      reportTrainF1 = []
      reportTrainKAPPA = []
      reportTestAUC = []
      reportTestAP = []
      reportTestF1 = []
      reportTestKAPPA = []
      startEpoch = 0
      minibatchCounterTrain = 0
      minibatchCounterTest = 0
      minibatchReportNr = 0
  
      if continueComputations:
          exec(open(methodPath + 'step1Load.py').read(), globals())
      endEpoch = nrEpochs
      saveScript = methodPath + 'step1Save.py'
      if basicArchitecture == "selu":
          exec(open(methodPath + 'runEpochsSELU.py').read(), globals())
      elif basicArchitecture == "relu":
          exec(open(methodPath + 'runEpochsReLU.py').read(), globals())
      if saveComputations:
          exec(open(methodPath + 'step1Save.py').read(), globals())
  
      if basicArchitecture == "selu":
          exec(open(methodPath + 'runPredictSELU.py').read(), globals())
      else:
          exec(open(methodPath + 'runPredictReLU.py').read(), globals())
  
      saveFilename = savePrefix + ".finaltest.auc.pckl"
      saveFile = open(saveFilename, "wb")
      pickle.dump(sumTestAUC, saveFile)
  
      saveFilename = savePrefix + ".finaltrain.auc.pckl"
      saveFile = open(saveFilename, "wb")
      pickle.dump(sumTrainAUC, saveFile)
      saveFile.close()
  
      saveFilename = savePrefix + ".evalPredict.hdf5"
      saveFile = h5py.File(saveFilename, "w")
      saveFile.create_dataset('predictions', data=predDenseTest)
      saveFile.close()
  
      saveFilename = savePrefix + ".eval"
      np.savetxt(saveFilename + ".cmpNames", np.array(testSamples), fmt="%s")
      np.savetxt(saveFilename + ".targetNames", np.array(targetAnnInd.index.values), fmt="%s")
  
      dbgOutput.close()
