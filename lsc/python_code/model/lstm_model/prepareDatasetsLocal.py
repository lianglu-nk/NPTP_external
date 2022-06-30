#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
if not (denseOutputData is None):
  trainDenseOutput=denseOutputData[sampleAnnInd[trainSamples].values].copy()
  testDenseOutput=denseOutputData[sampleAnnInd[testSamples].values].copy()
  trainPos=(trainDenseOutput > 0.5).sum(axis=0)
  trainNeg=(trainDenseOutput < -0.5).sum(axis=0)

if not (sparseOutputData is None):
  trainSparseOutput=sparseOutputData[sampleAnnInd[trainSamples].values].copy()
  trainSparseOutputTransposed=trainSparseOutput.copy().T.tocsr()
  trainSparseOutputTransposed.sort_indices()
  testSparseOutput=sparseOutputData[sampleAnnInd[testSamples].values].copy()
  testSparseOutputTransposed=testSparseOutput.copy().T.tocsr()
  testSparseOutputTransposed.sort_indices()
  trainPos=np.array([np.sum(trainSparseOutputTransposed[x].data > 0.5) for x in range(trainSparseOutputTransposed.shape[0])])
  trainNeg=np.array([np.sum(trainSparseOutputTransposed[x].data < -0.5) for x in range(trainSparseOutputTransposed.shape[0])])

trainProp=trainPos/(trainPos+trainNeg)
trainBias=np.log(trainProp/(1.0-trainProp))
trainBias[np.logical_not(np.logical_and(trainPos>10, trainNeg>10))]=0.0
trainBias[:]=0.0





nrDenseFeatures=0
if not (denseInputData is None):
  trainDenseInput=denseInputData[denseSampleIndex[trainSamples].values].copy()
  testDenseInput=denseInputData[denseSampleIndex[testSamples].values].copy()
  nrDenseFeatures=trainDenseInput.shape[1]

  if normalizeLocalDense:
    trainDenseMean1=np.nanmean(trainDenseInput, 0)
    trainDenseStd1=np.nanstd(trainDenseInput, 0)+0.0001
    trainDenseInput=(trainDenseInput-trainDenseMean1)/trainDenseStd1
    trainDenseInput=np.tanh(trainDenseInput)
    trainDenseMean2=np.nanmean(trainDenseInput, 0)
    trainDenseStd2=np.nanstd(trainDenseInput, 0)+0.0001
    trainDenseInput=(trainDenseInput-trainDenseMean2)/trainDenseStd2
    
    testDenseInput=(testDenseInput-trainDenseMean1)/trainDenseStd1
    testDenseInput=np.tanh(testDenseInput)
    testDenseInput=(testDenseInput-trainDenseMean2)/trainDenseStd2
  
  trainDenseInput=np.nan_to_num(trainDenseInput)
  testDenseInput=np.nan_to_num(testDenseInput)



nrSparseFeatures=0
if not (sparseInputData is None):
  trainSparseInput=sparseInputData[sparseSampleIndex[trainSamples].values].copy()
  featSel=trainSparseInput.getnnz(axis=0)/float(trainSparseInput.shape[0])> sparsenesThr
  mydenseInputData=sparseInputData[:, featSel].tocsr()
  
  trainDenseInput=mydenseInputData[sparseSampleIndex[trainSamples].values].A
  testDenseInput=mydenseInputData[sparseSampleIndex[testSamples].values].A
  nrDenseFeatures=trainDenseInput.shape[1]
  
  if normalizeLocalDense:
    trainDenseMean1=np.nanmean(trainDenseInput, 0)
    trainDenseStd1=np.nanstd(trainDenseInput, 0)+0.0001
    trainDenseInput=(trainDenseInput-trainDenseMean1)/trainDenseStd1
    trainDenseInput=np.tanh(trainDenseInput)
    trainDenseMean2=np.nanmean(trainDenseInput, 0)
    trainDenseStd2=np.nanstd(trainDenseInput, 0)+0.0001
    trainDenseInput=(trainDenseInput-trainDenseMean2)/trainDenseStd2
    
    testDenseInput=(testDenseInput-trainDenseMean1)/trainDenseStd1
    testDenseInput=np.tanh(testDenseInput)
    testDenseInput=(testDenseInput-trainDenseMean2)/trainDenseStd2
  
  trainDenseInput=np.nan_to_num(trainDenseInput)
  testDenseInput=np.nan_to_num(testDenseInput)



if ("lstmSmilesInputData" in globals()) and (not (lstmSmilesInputData is None)):
  trainSmilesLSTMInput=np.array(lstmSmilesInputData)[lstmSampleIndex[trainSamples].values].copy()
  testSmilesLSTMInput=np.array(lstmSmilesInputData)[lstmSampleIndex[testSamples].values].copy()
  trainGraphLSTMInput=np.array(lstmGraphInputData)[lstmSampleIndex[trainSamples].values].copy()
  testGrtaphLSTMInput=np.array(lstmGraphInputData)[lstmSampleIndex[testSamples].values].copy()

  
  trainLSTMSideOutputMACCS=maccsInputData[lstmSampleIndex[trainSamples].values]
  trainLSTMSideOutputECFP=ecfpInputData[lstmSampleIndex[trainSamples].values]
  trainLSTMSideOutputFCFP=fcfpInputData[lstmSampleIndex[trainSamples].values]
  
  testLSTMSideOutputMACCS=maccsInputData[lstmSampleIndex[testSamples].values]
  testLSTMSideOutputECFP=ecfpInputData[lstmSampleIndex[testSamples].values]
  testLSTMSideOutputFCFP=fcfpInputData[lstmSampleIndex[testSamples].values]
  
  trainLSTMSideOutputECFP=trainLSTMSideOutputECFP[:,np.argsort(-trainLSTMSideOutputECFP.sum(0).A[0])[0:128]].A
  trainLSTMSideOutputFCFP=trainLSTMSideOutputFCFP[:,np.argsort(-trainLSTMSideOutputFCFP.sum(0).A[0])[0:128]].A
  testLSTMSideOutputECFP=testLSTMSideOutputECFP[:,np.argsort(-testLSTMSideOutputECFP.sum(0).A[0])[0:128]].A
  testLSTMSideOutputFCFP=testLSTMSideOutputFCFP[:,np.argsort(-testLSTMSideOutputFCFP.sum(0).A[0])[0:128]].A

  trainLSTMSideOutput=np.hstack([trainLSTMSideOutputMACCS, trainLSTMSideOutputECFP, trainLSTMSideOutputFCFP])
  testLSTMSideOutput=np.hstack([testLSTMSideOutputMACCS, testLSTMSideOutputECFP, testLSTMSideOutputFCFP])
