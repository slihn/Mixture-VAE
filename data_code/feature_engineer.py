import numpy as np
import pandas as pd

def create_jump_feature(df, length):
    centered_mean = df['x'].rolling(window=length, min_periods=1).mean()
    centered_mean = centered_mean.reset_index(level=0, drop=True)
    df['centered_mean_' + str(length)] = centered_mean
    
    centered_std = df['x'].rolling(window=length, min_periods=1).std()
    centered_std = centered_std.reset_index(level=0, drop=True)
    df['centered_std_' + str(length)] = centered_std

    right_window = length // 2
    left_window = length - right_window
    right_mean = df['x'].rolling(window=right_window, min_periods=1).mean()
    right_std = df['x'].rolling(window=right_window, min_periods=1).std()
    left_mean = df['x'].rolling(window=left_window, min_periods=1).mean().shift(right_window)
    left_std = df['x'].rolling(window=left_window, min_periods=1).std().shift(right_window)
    
    right_mean = right_mean.reset_index(level=0, drop=True)
    right_std = right_std.reset_index(level=0, drop=True)
    left_mean = left_mean.reset_index(level=0, drop=True)
    left_std = left_std.reset_index(level=0, drop=True)

    df['right_mean_' + str(length)] = right_mean
    df['right_std_' + str(length)] = right_std
    df['left_mean_' + str(length)] = left_mean
    df['left_std_' + str(length)] = left_std

    return df

def create_feature(df):
    df['absolute_change'] = df['x'].diff().abs()
    df['previous_absolute_change'] = df['x'].diff().abs().shift(1)
    return df

def apply_feature_engineering(X):
    dfX = pd.DataFrame(X, columns=['x'])
    dfX = create_feature(dfX)
    for length in [6, 14]:
        dfX = create_jump_feature(dfX, length)
    #dfX = dfX.dropna().reset_index(drop=True)
    dfX = dfX.fillna(0).reset_index(drop=True)
    return dfX.values