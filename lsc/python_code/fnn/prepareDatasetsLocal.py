#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
if not (sparseOutputData is None):
    trainSparseOutput = sparseOutputData[sampleAnnInd[trainSamples].values].copy()
    trainSparseOutputTransposed = trainSparseOutput.copy().T.tocsr()
    trainSparseOutputTransposed.sort_indices()
    testSparseOutput = sparseOutputData[sampleAnnInd[testSamples].values].copy()
    testSparseOutputTransposed = testSparseOutput.copy().T.tocsr()
    testSparseOutputTransposed.sort_indices()
    trainPos = np.array(
        [np.sum(trainSparseOutputTransposed[x].data > 0.5) for x in range(trainSparseOutputTransposed.shape[0])])
    trainNeg = np.array(
        [np.sum(trainSparseOutputTransposed[x].data < -0.5) for x in range(trainSparseOutputTransposed.shape[0])])

trainProp = trainPos / (trainPos + trainNeg)
trainBias = np.log(trainProp / (1.0 - trainProp))
trainBias[np.logical_not(np.logical_and(trainPos > 10, trainNeg > 10))] = 0.0
trainBias[:] = 0.0


nrDenseFeatures = 0
if not (denseInputData is None):
    trainDenseInput = denseInputData[denseSampleIndex[trainSamples].values].copy()
    testDenseInput = denseInputData[denseSampleIndex[testSamples].values].copy()
    nrDenseFeatures = trainDenseInput.shape[1]

    if normalizeLocalDense:
        trainDenseMean1 = np.nanmean(trainDenseInput, 0)
        trainDenseStd1 = np.nanstd(trainDenseInput, 0) + 0.0001
        trainDenseInput = (trainDenseInput - trainDenseMean1) / trainDenseStd1
        trainDenseInput = np.tanh(trainDenseInput)
        trainDenseMean2 = np.nanmean(trainDenseInput, 0)
        trainDenseStd2 = np.nanstd(trainDenseInput, 0) + 0.0001
        trainDenseInput = (trainDenseInput - trainDenseMean2) / trainDenseStd2

        testDenseInput = (testDenseInput - trainDenseMean1) / trainDenseStd1
        testDenseInput = np.tanh(testDenseInput)
        testDenseInput = (testDenseInput - trainDenseMean2) / trainDenseStd2

    trainDenseInput = np.nan_to_num(trainDenseInput)
    testDenseInput = np.nan_to_num(testDenseInput)

nrSparseFeatures = 0

if ("graphInputData" in globals()) and (not (graphInputData is None)):
    trainGraphInput = graphInputData[graphSampleIndex[trainSamples].values].copy()
    testGraphInput = graphInputData[graphSampleIndex[testSamples].values].copy()
