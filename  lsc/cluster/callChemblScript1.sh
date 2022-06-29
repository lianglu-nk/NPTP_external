#implementation similar to http://www.bioinf.jku.at/research/lsc/
#with the following copyright information:
#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)
#!/bin/bash

out_file_name=$1
inputfliename=$2
sdfFile=$3

workdir=$(cd $(dirname $0); pwd)
rawDataDir=(${workdir//cluster/})

echo $1
echo $2
echo $3

morganDataDir=$rawDataDir$out_file_name

mkdir $morganDataDir
mkdir $morganDataDir/Morgan
mkdir $morganDataDir/Morgan/ECFC4
$workdir/chemblScript1.sh ECFC 2 DAYLIGHT_INVARIANT_RING ${rawDataDir}$inputfliename/$sdfFile $morganDataDir/Morgan/ECFC4 chembl_id LIBSVM_SPARSE_FREQUENCY


dirName=$morganDataDir/Morgan/ECFC4
outFile=ECFC4.fpf
echo `whoami`
# rm $morganDataDir/Morgan/$outFile
for i in `ls $dirName/myout*.res`; do
  echo $i
  head -n`wc -l $i |cut -f 1 -d " "` $i >> $morganDataDir/Morgan/$outFile
done

cat ${rawDataDir}$inputfliename/$sdfFile|grep -A1 chembl_id|grep -v chembl_id|grep -v \\-\\- > $morganDataDir/Morgan/SampleIdTable.txt
