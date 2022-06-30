#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr
basicArchitecture=hyperParams.iloc[paramNr].basicArchitecture
currentLR=hyperParams.iloc[paramNr].learningRate
dropout=hyperParams.iloc[paramNr].dropout
graphLayers=hyperParams.iloc[paramNr].graphLayers
denseLayers=hyperParams.iloc[paramNr].denseLayers





import deepchem.feat
import deepchem.feat.mol_graphs
import deepchem.metrics




exec(open(methodPath+'graphModels.py').read(), globals())



seed=123
if basicArchitecture=="Weave":
  model=MyWeaveTensorGraph(n_tasks=nrOutputTargets, n_atom_feat=75, n_pair_feat=14, n_hidden=graphLayers, n_graph_feat=denseLayers, batch_size=batchSize, dropout=dropout, learning_rate=currentLR, use_queue=False, random_seed=seed, mode='classification', configproto=gpu_options, verbose=False, mycapacity=5)
  singleFunc=weaveFunc
  batchFunc=weaveInput

if not model.built:
  model.build()
  with model._get_tf("Graph").as_default():
    updateOps=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

