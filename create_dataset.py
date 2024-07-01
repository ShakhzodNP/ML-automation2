import numpy as np
import pandas as pd
import random
import os

def get_df(n=60):
    url = 'https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv'
    df = pd.read_csv(url)
    return df


def main():
    if not(os.path.isdir('./test')):
        os.mkdir('./test')
    if not(os.path.isdir('./train')):
        os.mkdir('./train')
    df = get_df()
    test = df.sample(frac=0.2)
    train = df.drop(test.index,axis=0)
    test.to_csv('./test/df1.csv')
    train.to_csv('./train/df1.csv')

if __name__ == "__main__":
    main()