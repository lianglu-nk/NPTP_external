/*
Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
Copyright (C) 2018 Andreas Mayr
*/



#ifdef multiproc
#include <omp.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <list>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sstream>
#include <vector>
#include <unistd.h>

using namespace std;

vector<string> splitString(const string& s)
{
	vector<string> ans;
	int len = s.length();
	if (len == 0) return ans;
	for (int i = 0; i < len;){
		int pos = s.find("cluster", i);
		if (pos != string::npos){
			if (pos == i){
				i = pos + 1;
				continue;
			}
			else{
				string strTemp = s.substr(i, pos - i);
				ans.push_back(strTemp);
				i = pos + 1;
			}
		}
		else{
			string strTemp = s.substr(i, len - i);
			ans.push_back(strTemp);
			break;
		}
	}
	return ans;
}


int main(int argc, char** argv) {
  if(sizeof(long)<8) {
    printf("This program is optimized for machines with at least 8 byte longs!  The program will terminate!\n");
    return -1;
  }
  
  string path=getcwd(NULL,0);
  
  string progName(argv[0]);
  vector<string> ans = splitString(path);
  
  
  string projectName(argv[1]);
  
  //string basePathname("../../"); 
  string basePathname(ans[0]);
  
  /*for (int i = 0; i < ans.size(); ++i){
    string m(ans[i]);
    printf(m.c_str());
  }*/

 
  basePathname=basePathname+"/"+projectName+"/";
  
  
  
  if(argc!=3) {
    printf("Usage: genDirStructure projectName rawDataFolder\n");
    return -1;
  }

  
  string projectPathname=basePathname+"out_put_file/";
  if(mkdir(projectPathname.c_str(), S_IRWXU)==-1) {
    fprintf(stderr, "The directory '%s' either already exists or it is an invalid directory!\n", projectPathname.c_str());
    exit(-1);
  }
  string chemPathname=projectPathname+"chemFeatures/";
  string clusterPathname=chemPathname+"cl";
  string schemPathname=chemPathname+"s";
  string trainPathname=projectPathname+"train";
  string destSampleIdFilename=chemPathname+"SampleIdTable.txt";
  
  string morganDataFoldername(argv[2]);
  //string morganDataPathname("../../");
  string morganDataPathname=basePathname+morganDataFoldername+"/";
  string sourceSampleIdFilename=morganDataPathname+"SampleIdTable.txt";
  
  
  
  mkdir(chemPathname.c_str(), S_IRWXU);
  mkdir(clusterPathname.c_str(), S_IRWXU);
  mkdir(schemPathname.c_str(), S_IRWXU);
  mkdir(trainPathname.c_str(), S_IRWXU);


  
  FILE *fd1=fopen(sourceSampleIdFilename.c_str(), "rb");
  FILE *fd2=fopen(destSampleIdFilename.c_str(), "wb");

  size_t l1;
  unsigned char buffer[8192]; 

  while((l1 = fread(buffer, 1L, sizeof(unsigned char)*8192L, fd1)) > 0) {
    fwrite(buffer, 1, l1, fd2);
  }
  
  fclose(fd1);
  fclose(fd2);

  printf(progName.c_str());
  printf(" terminated successfully!\n");
  
  return 0;
}
