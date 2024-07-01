from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
import pickle



def main():
    filename = 'model.pkl'
    model = pickle.load(open(filename, 'rb'))
    dir = './test'
    df = None
    if os.path.isdir(dir):
        for i in os.listdir(dir):
            if i.find('.csv') != -1:
                df = pd.concat([df,pd.read_csv(dir+'/'+i,index_col=0)]) if df is not None else pd.read_csv(dir+'/'+i,index_col=0)
        X = df.drop(['Wine'],axis=1)
        y_pred = model.predict(X)
        print(model.score(X,df['Wine']))

if __name__ == "__main__":
    main()