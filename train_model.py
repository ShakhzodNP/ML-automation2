from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
import pickle

def main():
    model = RandomForestClassifier()
    dir = './train'
    df = None
    if os.path.isdir(dir):
        for i in os.listdir(dir):
            if i.find('.csv') != -1:
                df = pd.concat([df,pd.read_csv(dir+'/'+i,index_col=0)]) if df is not None else pd.read_csv(dir+'/'+i,index_col=0)
        y = df['Wine']
        X = df.drop(['Wine'],axis=1)
        model.fit(X,y)
        filename = 'model.pkl'
        pickle.dump(model, open(filename, 'wb'))

if __name__ == "__main__":
    main()