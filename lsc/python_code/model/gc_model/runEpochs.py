#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr

indMod=np.where(np.array([type(mychemblConvertedMols[x])==np.ndarray for x in trainGraphInput]))
trainDenseOutput[indMod,:]=0.0
uranium=singleFunc(rdkit.Chem.MolFromSmiles("[U]"))

for epoch in range(startEpoch, endEpoch):
  if epoch%50==0:
    if saveComputations:
      exec(open(saveScript).read(), globals())

  print("\n", file=dbgOutput)
  print("epoch: "+str(epoch)+"\n", file=dbgOutput)
  print("\n", file=dbgOutput)
  idxSamples=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(trainSamples)/batchSize)), shuffle=True).split(np.arange(len(trainSamples)))]
  for i in range(len(idxSamples)):
    #print(i)
    batchGraphX=mychemblConvertedMols[trainGraphInput[idxSamples[i]]]
    batchGraphX=np.array([uranium if type(x)==np.ndarray else x for x in batchGraphX])
    batchDenseY=trainDenseOutput[idxSamples[i]]
    extendSize=0
    if len(batchGraphX)<model.batch_size:
      extendSize=model.batch_size-len(batchGraphX)
      batchGraphX=np.append(batchGraphX, batchGraphX[0:extendSize])
      batchDenseY=np.vstack([batchDenseY, np.zeros_like(batchDenseY)[0:extendSize]])
    batchWeight=(np.abs(batchDenseY)>0.5).astype(np.integer)
    
    #batchInputSingle=[singleFunc(molX) for molX in batchGraphX]
    batchInput=batchFunc(model, batchGraphX)
    myfeedDict=batchInput
    myfeedDict[model.label]=(batchDenseY>0.5).astype(np.integer)
    myfeedDict[model.weights]=(np.abs(batchDenseY)>0.5).astype(np.integer)
    myfeedDict[model._training_placeholder]=1.0
    with model._get_tf("Graph").as_default():
      try:
        model.session.run([model._get_tf('train_op'), updateOps], feed_dict=myfeedDict)
      except:
        print("Error in Training!")
    
    
    
    minibatchCounterTrain=minibatchCounterTrain+1
    if minibatchCounterTrain==minibatchesPerReportTrain:
      if computeTrainPredictions:
        predDenseTrain=[]
        idxSamplesEval=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(trainSamples)/batchSize)), shuffle=False).split(np.arange(len(trainSamples)))]
        for j in range(len(idxSamplesEval)):
          batchGraphX=mychemblConvertedMols[trainGraphInput[idxSamplesEval[j]]]
          batchGraphX=np.array([uranium if type(x)==np.ndarray else x for x in batchGraphX])
          if compPerformanceTrain:
            batchDenseY=trainDenseOutput[idxSamplesEval[j]]
          extendSize=0
          if len(batchGraphX)<model.batch_size:
            extendSize=model.batch_size-len(batchGraphX)
            batchGraphX=np.append(batchGraphX, batchGraphX[0:extendSize])
          
          #batchInputSingle=[singleFunc(molX) for molX in batchGraphX]
          batchInput=batchFunc(model, batchGraphX)
          myfeedDict=batchInput
          myfeedDict[model._training_placeholder]=0.0
          with model._get_tf("Graph").as_default():
            predDenseTrain.append(model.session.run(model.outputs[0], feed_dict=myfeedDict)[0:(model.batch_size-extendSize)])
        
        predDenseTrain=np.vstack(predDenseTrain)
        if compPerformanceTrain:
          sumTrainAUC=np.array(utilsLib.calculateAUCs(trainDenseOutput, predDenseTrain))
        
        if compPerformanceTrain:
          reportTrainAUC.append(sumTrainAUC)
          print("\n", file=dbgOutput)
          print("Train Mean AUC: ", file=dbgOutput)
          print(np.nanmean(sumTrainAUC), file=dbgOutput)
          print("\n", file=dbgOutput)
          dbgOutput.flush()
        
        predDenseTrain=None
      
      
      minibatchCounterTrain=0
    
    
    
    minibatchCounterTest=minibatchCounterTest+1
    if minibatchCounterTest==minibatchesPerReportTest:
      if computeTestPredictions:
        predDenseTest=[]
        idxSamplesEval=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(testSamples)/batchSize)), shuffle=False).split(np.arange(len(testSamples)))]
        for j in range(len(idxSamplesEval)):
          batchGraphX=mychemblConvertedMols[testGraphInput[idxSamplesEval[j]]]
          batchGraphX=np.array([uranium if type(x)==np.ndarray else x for x in batchGraphX])
          if compPerformanceTrain:
            batchDenseY=testDenseOutput[idxSamplesEval[j]]
          extendSize=0
          if len(batchGraphX)<model.batch_size:
            extendSize=model.batch_size-len(batchGraphX)
            batchGraphX=np.append(batchGraphX, batchGraphX[0:extendSize])
          
          #batchInputSingle=[singleFunc(molX) for molX in batchGraphX]
          batchInput=batchFunc(model, batchGraphX)
          myfeedDict=batchInput
          myfeedDict[model._training_placeholder]=0.0
          with model._get_tf("Graph").as_default():
            predDenseTest.append(model.session.run(model.outputs[0], feed_dict=myfeedDict)[0:(model.batch_size-extendSize)])
        
        predDenseTest=np.vstack(predDenseTest)
        if compPerformanceTest:
          sumTestAUC=np.array(utilsLib.calculateAUCs(testDenseOutput, predDenseTest))
        
        if compPerformanceTest:
          reportTestAUC.append(sumTestAUC)
          print("\n", file=dbgOutput)
          print("Test Mean AUC: ", file=dbgOutput)
          print(np.nanmean(sumTestAUC), file=dbgOutput)
          print("\n", file=dbgOutput)
          dbgOutput.flush()
        
        predDenseTest=None
        
      
      
      
      minibatchCounterTest=0
      minibatchReportNr=minibatchReportNr+1
