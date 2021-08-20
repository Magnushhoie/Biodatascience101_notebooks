import pandas as pd
import numpy as np
import scipy as sc
from sklearn import metrics
import getpass
import glob
import os
from joblib import dump
import subprocess
import hashlib 

class competition_model_class:
    def __init__(self):
        data = pd.read_csv('data/test_set_proc__breast-cancer-wisconsin-data.csv')
        self.test_y = data.iloc[:,0]
        self.test_x = data.iloc[:,1:]
        self.results = {}
        self.best = {'idx':0, 'acc':0}
        
        self.results_folder = "/biodatascience_notebooks/competition/results"
        self.models_folder = "/biodatascience_notebooks/competition/models"
        self.user = getpass.getuser()
        
    def Submit(self, model, features ):
        num = self.number()
        self.results[num] = self.test_performance_module(model, features)
        if num <= 3:
            print('Your model {} / 3 is being submitted for the competition!'.format(num))
            self._submit_result(self.results[num], num)
            self._submit_model(self.results[num], num)
        else:
            print('You have exceded your 3 model submissions. This model will not be included in the competition.')
        if self.results[num]['Accuracy'] > self.best['acc']:
            self.best = {'idx':num, 'acc':self.results[num]['Accuracy']}
        
    def test_performance_module(self, model, features):
        y_hat = model.predict(self.test_x[features])
        y_true = self.test_y

        #Calculate accuracy
        acc = np.round(metrics.accuracy_score(y_true, y_hat), 10)

        # Next calculate the false true positive rates and precision
        precision = np.round(metrics.precision_score(y_true, y_hat, average='binary'), 10)
        recall = np.round(metrics.recall_score(y_true, y_hat), 10)

        hash_res = self._get_hash(y_hat, y_true)

        print("Accuracy :", acc)
        print("Precision :", precision)
        print("Recall :", recall)
        

        return {'Accuracy':acc, 

                'Recall':recall,
                'Precision':precision,
                
                'Features':features,
                'Params':str(model),
                'Hash': hash_res,
                'Model':model
               }
    
    def number(self):
        num = len(glob.glob(os.path.join(self.results_folder, '{}_*.txt'.format(self.user))))
        return num + 1
    
    
    def get_best_model(self):
        for i in ['Accuracy', 'Recall', 'Precision', 'Features']:
            print(i, ' : ', self.results[self.best['idx']][i])
        return self.results[self.best['idx']]['Model']
    
    def _get_hash(self, y_hat, y_true):
        
        test = str(np.round(metrics.accuracy_score(y_true, y_hat), 10)) +str(np.round(metrics.recall_score(y_true, y_hat), 10))
        return hashlib.md5(test.encode()).hexdigest()
        
        
        
        
        
    
    def _submit_result(self, result, num):
        result['Features'] = str(result['Features']).strip('[]')
        
        pd.DataFrame(result, index = [1]).T.loc[['Accuracy', 
                                         'Recall', 'Precision', 'Features', 'Params', 'Hash']].to_csv(os.path.join(self.results_folder,'{}_{}.txt').format(self.user, num), sep = '\t', header = False)
        subprocess.call(['chmod', '0400', os.path.join(self.results_folder,'{}_{}.txt').format(self.user, num)])
        
    
    def _submit_model(self, result, num):
        dump(result['Model'], os.path.join(self.models_folder,'{}_{}.joblib').format(self.user, num), compress=9) 
        subprocess.call(['chmod', '0400', os.path.join(self.models_folder,'{}_{}.joblib').format(self.user, num)])
        
