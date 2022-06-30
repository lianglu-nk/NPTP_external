#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr

library(dplyr)

Args= commandArgs(trailingOnly=TRUE)
projectName=Args[1]
targetName=Args[2]

clusterInfo="cl1"
clusterVersion="clusterMinFull"

c_dir=getwd()
basePathname=strsplit(c_dir,"cluster")[[1]][1]

projectPathname=paste0(basePathname, "/",projectName,"/out_put_file");

chemPathname=file.path(projectPathname, "chemFeatures");
clusterPathname=file.path(chemPathname, "cl");
schemPathname=file.path(chemPathname, "s");
trainPathname=file.path(projectPathname, "train");

sampleIdFilename=file.path(chemPathname, "SampleIdTable.txt")

clusterSampleFilename=file.path(clusterPathname, paste0(clusterInfo, ".info"))
countFilename=file.path(clusterPathname, paste0("count", ".info"))
targetFilename=file.path(clusterPathname, paste0("tar", ".info"))
tarFilename=file.path(clusterPathname, paste0("target", ".info"))


seed_number = 12345L
cluster_file=paste0(clusterPathname,'/',clusterVersion,"/clustering_70.txt")
#rawClusterTable=read.table("/home/jianping/mydata/trgpred/chembl20/chemFeatures/cl/clusterMinFull/clustering_70.txt", stringsAsFactors=FALSE, sep=",")
rawClusterTable=read.table(cluster_file, stringsAsFactors=FALSE, sep=",")
target_file=paste0(trainPathname,"/",targetName,".csv")
target_mol_table = read.table(target_file, stringsAsFactors=FALSE, sep=",")
names(target_mol_table)=c("label","mol","target")
clusterSizes=table(rawClusterTable[,1])
for(i in 1:seed_number){
  set.seed(i)
  clusterSizes=clusterSizes[sample(length(clusterSizes))]
  clusterSizes=clusterSizes[order(clusterSizes)]
  if((length(clusterSizes)%%3L)>0L) {
   clusterAssignment=c(rep(0L:2L, length(clusterSizes)%/%3L), (0L:2L)[1L:(length(clusterSizes)%%3L)])
  } else {
    clusterAssignment=rep(0L:2L, length(clusterSizes)%/%3L)
  }
  names(clusterAssignment)=names(clusterSizes)
  newclusterAssignment=clusterAssignment[as.character(rawClusterTable[,1])]
  names(newclusterAssignment)=NULL
  newClusterTable=rawClusterTable
  newClusterTable[,1]=newclusterAssignment
  names(newClusterTable)=c("cluster","mol")
  newClustertarget = merge(target_mol_table,newClusterTable,by="mol",all=FALSE)[,-1]
  newClustertarget_count_tmp =group_by(newClustertarget,target,cluster)
  newClustertarget_count =summarise(newClustertarget_count_tmp,n=n_distinct(label))
#  write.table(newClustertarget, file=tarFilename, row.names=FALSE, col.names=FALSE, quote=FALSE)
  newClustertarget
#  write.table(newClustertarget_count_tmp, file=targetFilename, row.names=FALSE, col.names=FALSE, quote=FALSE)
  newClustertarget_count_tmp
#  write.table(newClustertarget_count, file=countFilename, row.names=FALSE, col.names=FALSE, quote=FALSE)
  newClustertarget_count
  if(min(newClustertarget_count[,3])==2){
    write.table(newClusterTable, file=clusterSampleFilename, row.names=FALSE, col.names=FALSE, quote=FALSE)
    print(i)
    break
  }

}







