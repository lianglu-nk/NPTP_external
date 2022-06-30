#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
import sys

trainLSTMOutput=np.hstack([(trainDenseOutput+1)/2.0, trainLSTMSideOutput])

for epoch in range(startEpoch, endEpoch):
  print(epoch)
  
  if epoch%50==0:
    if saveComputations:
      exec(open(saveScript).read(), globals())
  
  print("\n", file=dbgOutput)
  print("epoch: "+str(epoch)+"\n", file=dbgOutput)
  print("\n", file=dbgOutput)
  
  
  
  idxSamples=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(trainSamples)/batchSize)), shuffle=True).split(np.arange(len(trainSamples)))]
  for i in range(len(idxSamples)):
    batchX=np.array([myOneHot(myRandomize(x2), oneHot, otherInd, pad_len=seq_length) if x1!='' else myOneHot('', oneHot, otherInd, pad_len=seq_length) for x1, x2 in zip(trainSmilesLSTMInput[idxSamples[i]], trainGraphLSTMInput[idxSamples[i]])])
    batchY=trainLSTMOutput[idxSamples[i]]
    model.train_on_batch(batchX, batchY)
    
    
    minibatchCounterTrain=minibatchCounterTrain+1
    if minibatchCounterTrain==minibatchesPerReportTrain:
      if computeTrainPredictions:
        predDenseTrain=[]
        idxSamplesEval=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(trainSamples)/batchSize)), shuffle=False).split(np.arange(len(trainSamples)))]
        for j in range(len(idxSamplesEval)):
          batchX=np.array([myOneHot(x, oneHot, otherInd, pad_len=seq_length) for x in trainSmilesLSTMInput[idxSamplesEval[j]]])
          batchY=trainLSTMOutput[idxSamplesEval[j]]
          predDenseTrain.append(model.predict_on_batch(batchX))
        
        predDenseTrain=np.vstack(predDenseTrain)
        if compPerformanceTrain:
          sumTrainAUC=np.array(utilsLib.calculateAUCs(trainDenseOutput, predDenseTrain[:,0:nrOutputTargets]))
          sumTrainAUCCheck=np.array(utilsLib.calculateAUCs(trainLSTMSideOutput*2-1, predDenseTrain[:,nrOutputTargets:]))
        
        if compPerformanceTrain:
          reportTrainAUC.append(sumTrainAUC)
          print("\n", file=dbgOutput)
          print("Train Mean AUC: ", file=dbgOutput)
          print(np.nanmean(sumTrainAUC), file=dbgOutput)
          print("\n", file=dbgOutput)
          print("Train Check Mean AUC: ", file=dbgOutput)
          print(np.nanmean(sumTrainAUCCheck), file=dbgOutput)
          print("\n", file=dbgOutput)
        
        predDenseTrain=None
      
      
      minibatchCounterTrain=0
    
    
    
    minibatchCounterTest=minibatchCounterTest+1
    if minibatchCounterTest==minibatchesPerReportTest:
      if computeTestPredictions:
        predDenseTest=[]
        idxSamplesEval=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(testSamples)/batchSize)), shuffle=False).split(np.arange(len(testSamples)))]
        for j in range(len(idxSamplesEval)):
          batchX=np.array([myOneHot(x, oneHot, otherInd, pad_len=seq_length) for x in testSmilesLSTMInput[idxSamplesEval[j]]])
          predDenseTest.append(model.predict_on_batch(batchX))
        
        predDenseTest=np.vstack(predDenseTest)
        if compPerformanceTest:
          sumTestAUC=np.array(utilsLib.calculateAUCs(testDenseOutput, predDenseTest[:,0:nrOutputTargets]))
          sumTestAUCCheck=np.array(utilsLib.calculateAUCs(testLSTMSideOutput*2-1, predDenseTest[:,nrOutputTargets:]))
        
        if compPerformanceTest:
          reportTestAUC.append(sumTestAUC)
          print("\n", file=dbgOutput)
          print("Test Mean AUC: ", file=dbgOutput)
          print(np.nanmean(sumTestAUC), file=dbgOutput)
          print("\n", file=dbgOutput)
          print("Test Check Mean AUC: ", file=dbgOutput)
          print(np.nanmean(sumTestAUCCheck), file=dbgOutput)
          print("\n", file=dbgOutput)
          dbgOutput.flush()
        
        predDenseTest=None
        
      
      
      
      minibatchCounterTest=0
      minibatchReportNr=minibatchReportNr+1
