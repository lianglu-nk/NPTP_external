#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
#!/bin/bash

optionAlgorithm=$1
optionSearchDepth=$2
optionAtomType=$3
optionInputFile=$4
optionOutputDir=$5
optionLabel=$6
optionOutputFormat=$7

echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
echo $7

workdir=$(cd $(dirname $0); pwd)
rawDataDir=(${workdir//cluster/})

nrLines=`wc -l $optionInputFile|cut -f 1 -d " "`

cp $optionInputFile "$optionOutputDir/inputFile.sdf"
inputFile="$optionOutputDir/inputFile.sdf"
fileNr=1
outputFile="$optionOutputDir/myout$fileNr.res"
java -Xmx4g -XX:hashCode=5 -jar ${rawDataDir}programs/jcompoundmapper-code-r55/bin/jCMapperCLI.jar -c $optionAlgorithm -d $optionSearchDepth -hs 2147483647 -a $optionAtomType -f $inputFile -o $outputFile  -l $optionLabel -ff $optionOutputFormat
cnt=1

oldFile=$outputFile
fileNr=`expr $fileNr + 1`
outputFile="$optionOutputDir/myout$fileNr.res"
newInchi=`tail -n1 $oldFile|cut -f 1 -d " "`

while [ 1 -gt 0 ]; do
  echo $outputFile
  cnt=1
  while [ \( ! -e "$outputFile" \) -o \( ! -s "$outputFile" \) ]; do
    echo $cnt
    cat -n $optionInputFile | grep -E "($newInchi|\\$\\$\\$\\$)" | grep -A$cnt $newInchi|tail -n1|cut -f1 > $optionOutputDir/lineFile
    myLine=`cat $optionOutputDir/lineFile`
    if [ -z "$myLine" ]; then
      exit
    fi
    catLines=`expr $nrLines - $myLine`
    echo $catLines
    if [ $catLines -lt 1 ]; then
      exit
    fi
    tail -n $catLines $optionInputFile > $inputFile
    java -Xmx4g -XX:hashCode=5 -jar $HOME/myprogs/jcompoundmapper-code-r55/bin/jCMapperCLI.jar -c $optionAlgorithm -d $optionSearchDepth -hs 2147483647 -a $optionAtomType -f $inputFile -o $outputFile -l $optionLabel -ff $optionOutputFormat
    cnt=`expr $cnt + 1`
  done
  oldFile=$outputFile
  fileNr=`expr $fileNr + 1`
  outputFile="$optionOutputDir/myout$fileNr.res"
  newInchi=`tail -n1 $oldFile|cut -f 1 -d " "`
done
