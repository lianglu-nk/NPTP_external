#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
if computeTestPredictions:
  if compPerformanceTest:
    saveFilename=savePrefix+".test.auc.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(reportTestAUC, saveFile)
    saveFile.close()

if computeTrainPredictions:
  if compPerformanceTrain:
    saveFilename=savePrefix+".train.auc.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(reportTrainAUC, saveFile)
    saveFile.close()

saveFilename=savePrefix+".trainInfo.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(epoch, saveFile)
pickle.dump(minibatchCounterTrain, saveFile)
pickle.dump(minibatchCounterTest, saveFile)
pickle.dump(minibatchReportNr, saveFile)
saveFile.close()

saveFilename=savePrefix+".trainModel"
model.save(saveFilename)

saveFilename=savePrefix+"."+finMark+".pckl"
saveFile=open(saveFilename, "wb")
finNr=0
pickle.dump(finNr, saveFile)
saveFile.close()
