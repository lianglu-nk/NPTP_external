#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
if nrSparseFeatures > 0.5:
    session.run(sparseMeanWSparseOp.op)

for epoch in range(startEpoch, endEpoch):
    print(epoch)

    if np.any(session.run(checkNA)):
        print("\n", file=dbgOutput)
        print("Attention: NaN detected! Computation stopped!\n", file=dbgOutput)
        saveFilename = savePrefix + ".error"
        saveFile = open(saveFilename, "wb")
        errorNr = 0
        pickle.dump(errorNr, saveFile)
        saveFile.close()
        break

    if epoch % 50 == 0:
        if saveComputations:
            exec(open(saveScript).read(), globals())

    print("\n", file=dbgOutput)
    print("epoch: " + str(epoch) + "\n", file=dbgOutput)
    print("\n", file=dbgOutput)
    idxSamples = [arr[1] for arr in
                  sklearn.model_selection.KFold(n_splits=int(math.ceil(len(trainSamples) / batchSize)),
                                                shuffle=True).split(np.arange(len(trainSamples)))]
    for i in range(len(idxSamples)):

        myfeedDict = {
            inputDropout: currentIDropout,
            hiddenDropout: currentDropout,
            lrGeneral: currentLR,
            l2PenaltyWeight: currentL2Penalty,
            l2PenaltyBias: 0.0,
            l1PenaltyWeight: currentL1Penalty,
            l1PenaltyBias: 0.0,
            mom: currentMom,
            is_training: False
        }

        if nrDenseFeatures > 0.5:
            batchDenseX = trainDenseInput[idxSamples[i]]
            myfeedDict.update({
                xDenseData: batchDenseX
            })

        if nrSparseFeatures > 0.5:
            batchSparseX = trainSparseInput[idxSamples[i], :].copy()
            batchSparseX.sort_indices()
            nonzx = batchSparseX.nonzero()
            valnonzx = (batchSparseX)[nonzx[0], nonzx[1]]
            myfeedDict.update({
                xIndices: np.int64(np.vstack(nonzx).T),
                xValues: valnonzx.A.flatten(),
                xDim: [len(idxSamples[i]), batchSparseX.shape[1]]
            })

        if useDenseOutputNetTrain:
            batchDenseY = trainDenseOutput[idxSamples[i]]
            myfeedDict.update({
                yDenseData: batchDenseY
            })
            session.run(optimizerDense, feed_dict=myfeedDict)

        if not useDenseOutputNetTrain:
            batchSparseY = trainSparseOutput[idxSamples[i]].copy()
            batchSparseY.sort_indices()
            nonzy = batchSparseY.nonzero()
            valnonzy = (batchSparseY)[nonzy[0], nonzy[1]]
            myfeedDict.update({
                yIndices: np.int64(np.vstack(nonzy).T),
                yValues: valnonzy.A.flatten(),
                yDim: [len(idxSamples[i]), batchSparseY.shape[1]]
            })
            session.run(optimizerSparse, feed_dict=myfeedDict)

        if nrSparseFeatures > 0.5:
            if normalizeGlobalSparse or normalizeLocalSparse:
                session.run(sparseMeanWSparseOp.op)

        session.run(scalePredictId, feed_dict={inputDropout: currentIDropout, hiddenDropout: currentDropout})
        session.run(scalePredictHd, feed_dict={inputDropout: currentIDropout, hiddenDropout: currentDropout})

        minibatchCounterTrain = minibatchCounterTrain + 1
        if (minibatchCounterTrain == minibatchesPerReportTrain) and (not np.any(session.run(checkNA))):
            if computeTrainPredictions:
                predDenseTrain = []
                if not (sparseOutputData is None):
                    predSparseTrain = trainSparseOutput.copy().astype(np.float32)
                    predSparseTrain.data[:] = -1
                idxSamplesEval = [arr[1] for arr in
                                  sklearn.model_selection.KFold(n_splits=int(math.ceil(len(trainSamples) / batchSize)),
                                                                shuffle=False).split(np.arange(len(trainSamples)))]
                for j in range(len(idxSamplesEval)):

                    myfeedDict = {
                        inputDropout: 0.0,
                        hiddenDropout: 0.0,
                        is_training: False
                    }

                    if nrDenseFeatures > 0.5:
                        batchDenseX = trainDenseInput[idxSamplesEval[j]]
                        myfeedDict.update({
                            xDenseData: batchDenseX
                        })

                    if nrSparseFeatures > 0.5:
                        batchSparseX = trainSparseInput[idxSamplesEval[j], :].copy()
                        batchSparseX.sort_indices()
                        nonzx = batchSparseX.nonzero()
                        valnonzx = (batchSparseX)[nonzx[0], nonzx[1]]
                        myfeedDict.update({
                            xIndices: np.int64(np.vstack(nonzx).T),
                            xValues: valnonzx.A.flatten(),
                            xDim: [len(idxSamplesEval[j]), batchSparseX.shape[1]]
                        })

                    if useDenseOutputNetPred:
                        if compPerformanceTrain:
                            batchDenseY = trainDenseOutput[idxSamplesEval[j]]
                        predDenseTrain.append(session.run(predNetwork, feed_dict=myfeedDict))

                    if not useDenseOutputNetPred:
                        if compPerformanceTrain:
                            batchSparseY = trainSparseOutput[idxSamplesEval[j]].copy()
                            batchSparseY.sort_indices()
                            nonzy = batchSparseY.nonzero()
                            valnonzy = (batchSparseY)[nonzy[0], nonzy[1]]
                            myfeedDict.update({
                                yIndices: np.int64(np.vstack(nonzy).T)
                            })
                            predSparseTrain[min(idxSamplesEval[j]) + nonzy[0], nonzy[1]] = session.run(
                                predNetworkSparse, feed_dict=myfeedDict)

                if not useDenseOutputNetPred:
                    if compPerformanceTrain:
                        predSparseTrainTransposed = predSparseTrain.T.tocsr()
                        predSparseTrain = None
                        predSparseTrainTransposed.sort_indices()
                        sumTrainAUC = np.array(
                            utilsLib.calculateSparseAUCs(trainSparseOutputTransposed, predSparseTrainTransposed))
                        predSparseTrainTransposed = None

                if compPerformanceTrain:
                    reportTrainAUC.append(sumTrainAUC)
                    print("\n", file=dbgOutput)
                    print("Train Mean AUC: ", file=dbgOutput)
                    print(np.nanmean(sumTrainAUC), file=dbgOutput)
                    print("\n", file=dbgOutput)
                    dbgOutput.flush()

        if minibatchCounterTrain == minibatchesPerReportTrain:
            minibatchCounterTrain = 0

        minibatchCounterTest = minibatchCounterTest + 1
        if (minibatchCounterTest == minibatchesPerReportTest) and (not np.any(session.run(checkNA))):
            if computeTestPredictions:
                predDenseTest = []
                if not (sparseOutputData is None):
                    predSparseTest = testSparseOutput.copy().astype(np.float32)
                    predSparseTest.data[:] = -1

                idxSamplesEval = [arr[1] for arr in
                                  sklearn.model_selection.KFold(n_splits=int(math.ceil(len(testSamples) / batchSize)),
                                                                shuffle=False).split(np.arange(len(testSamples)))]
                for j in range(len(idxSamplesEval)):

                    myfeedDict = {
                        inputDropout: 0.0,
                        hiddenDropout: 0.0,
                        is_training: False
                    }

                    if nrDenseFeatures > 0.5:
                        batchDenseX = testDenseInput[idxSamplesEval[j]]
                        myfeedDict.update({
                            xDenseData: batchDenseX
                        })

                    if nrSparseFeatures > 0.5:
                        batchSparseX = testSparseInput[idxSamplesEval[j], :].copy()
                        batchSparseX.sort_indices()
                        nonzx = batchSparseX.nonzero()
                        valnonzx = (batchSparseX)[nonzx[0], nonzx[1]]
                        myfeedDict.update({
                            xIndices: np.int64(np.vstack(nonzx).T),
                            xValues: valnonzx.A.flatten(),
                            xDim: [len(idxSamplesEval[j]), batchSparseX.shape[1]]
                        })

                    if useDenseOutputNetPred:
                        if compPerformanceTest:
                            batchDenseY = testDenseOutput[idxSamplesEval[j]]
                        predDenseTest.append(session.run(predNetwork, feed_dict=myfeedDict))

                    if not useDenseOutputNetPred:
                        if compPerformanceTest:
                            batchSparseY = testSparseOutput[idxSamplesEval[j]].copy()
                            batchSparseY.sort_indices()
                            nonzy = batchSparseY.nonzero()
                            valnonzy = (batchSparseY)[nonzy[0], nonzy[1]]
                            myfeedDict.update({
                                yIndices: np.int64(np.vstack(nonzy).T)
                            })
                            predSparseTest[min(idxSamplesEval[j]) + nonzy[0], nonzy[1]] = session.run(predNetworkSparse,
                                                                                                      feed_dict=myfeedDict)

                if not useDenseOutputNetPred:
                    if compPerformanceTest:
                        predSparseTestTransposed = predSparseTest.copy().T.tocsr()
                        predSparseTest = None
                        predSparseTestTransposed.sort_indices()
                        sumTestAUC = np.array(
                            utilsLib.calculateSparseAUCs(testSparseOutputTransposed, predSparseTestTransposed))
                        predSparseTestTransposed = None

                if compPerformanceTest:
                    reportTestAUC.append(sumTestAUC)
                    print("\n", file=dbgOutput)
                    print("Test Mean AUC: ", file=dbgOutput)
                    print(np.nanmean(sumTestAUC), file=dbgOutput)
                    print("\n", file=dbgOutput)
                    dbgOutput.flush()

        if minibatchCounterTest == minibatchesPerReportTest:
            minibatchCounterTest = 0
            minibatchReportNr = minibatchReportNr + 1

        session.run(scaleTrainId, feed_dict={inputDropout: currentIDropout, hiddenDropout: currentDropout})
        session.run(scaleTrainHd, feed_dict={inputDropout: currentIDropout, hiddenDropout: currentDropout})
