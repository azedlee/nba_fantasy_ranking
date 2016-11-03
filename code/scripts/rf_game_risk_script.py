import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np

with open('/home/ubuntu/modeling/pickle_input/trainX_game_risk.pickle', 'rb') as f:
    trainX = pickle.load(f)
with open('/home/ubuntu/modeling/pickle_input/trainY_game_risk.pickle', 'rb') as f:
    trainY = pickle.load(f)

forest = RandomForestClassifier()

params = {'max_depth':[2,3,4,5,6,None], 
          'max_features':['auto'],
          'min_samples_split':[2,4,8,16,32,64,128,256],
          'n_estimators':[500],
          'criterion': ['gini']
         }

estimator_rfr = GridSearchCV(forest, params, n_jobs=-1,  cv=5, verbose=1) 

model = estimator_rfr.fit(trainX, trainY)

with open('/home/ubuntu/modeling/pickle_output/model_game_risk.pickle', 'wb') as f:
	pickle.dump(model, f)