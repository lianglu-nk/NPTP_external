#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
if continueComputations:
  if computeTestPredictions:
    if compPerformanceTest:
      saveFilename=savePrefix+".test.auc.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        reportTestAUC=pickle.load(saveFile)
        saveFile.close()

  if computeTrainPredictions:
    if compPerformanceTrain:
      saveFilename=savePrefix+".train.auc.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        reportTrainAUC=pickle.load(saveFile)
        saveFile.close()

saveFilename=savePrefix+".trainInfo.pckl"
if os.path.isfile(saveFilename):
  saveFile=open(saveFilename, "rb")
  startEpoch=pickle.load(saveFile)
  minibatchCounterTrain=pickle.load(saveFile)
  minibatchCounterTest=pickle.load(saveFile)
  minibatchReportNr=pickle.load(saveFile)
  saveFile.close()

saveFilename=savePrefix+".trainModel"
if os.path.isfile(saveFilename):
  saveFilename=savePrefix+".trainModel"
  #model=load_model(saveFilename)
  model.load_weights(saveFilename)
