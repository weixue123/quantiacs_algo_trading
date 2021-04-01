import os
import pandas as pd
from arch import arch_model
import pickle
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys

from utils.data_loader import load_processed_data
from systems.systems_util import get_futures_list

class GARCH:
    def __init__(self, future):
        self.future = future
    
    @staticmethod
    def get_training_data(ticker):
        data = load_processed_data(ticker)
        data = data.loc[:"2020-12-31"]
        data = data.iloc[:-1]   
        data = data[['CLOSE']]
        # Pre-process
        log_data = np.log(data[['CLOSE']])
        log_data = np.array(log_data['CLOSE'])
        diff_log_data = np.diff(log_data) 
        
        return diff_log_data
    
    def fit_model(self, p, q, params):
        pass
    
    def build_model(self, save=False):
        AIC = {}
        data = GARCH.get_training_data(self.future) * 1000
        best_aic = sys.maxsize
        optimal_p = 0
        optimal_q = 0
        optimal_params = None
        optimal_model = None
        for p in range(1,10):
            for q in range(1,10):
                model = arch_model(data, p=p, q=q, vol="GARCH")
                try:
                    fit_model = model.fit()
                except Exception as e:
                    print(f"Exception {e} for p: {p} and q: {q}")
                    continue
                aic = fit_model.aic
                if aic < best_aic:
                    best_aic = aic
                    optimal_p = p
                    optimal_q = q
                    optimal_params = list(fit_model.params)
                    optimal_model = model
        all_params = {"p": optimal_p, "q": optimal_q, "params": optimal_params}
        if save:
            self.save_params(all_params)
            self.save_model(optimal_model)
        return all_params, optimal_model
        
    def save_params(self, params):
        dire = f"./params/"
        os.makedirs(os.path.dirname(dire), exist_ok=True)
        with open(f'{dire}/{self.future}_params.txt', 'w') as f:
            json.dump(params, f, ensure_ascii=False)
            print(f"Saved parameters for {self.future}")
            
    def save_model(self, model):
        dire = f"./models/"
        pickle_out = open(f"{dire}/{self.future}.pickle", "wb")
        pickle.dump(model, pickle_out)
        pickle_out.close()
        print(f"Saved model for {self.future}")