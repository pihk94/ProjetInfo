import pandas as pd
import numpy as np 

def CRP(df,start,end):
    """
        Input : 
            df : dataframe
                type : dataframe
            start : debut de période
                type : datetime
            end : fin de période
                type : datetime
        Output : 
            liste des rendements de la stratégie

        Stratégie : Basé sur https://www.cs.princeton.edu/courses/archive/spring08/cos511/scribe_notes/0428.pdf
            A chaque début de journée, on revend tout et on rachete dans les proportions choisis uniforément au départ (1/N).
    """
    poids = [1/len(df.columns) for i in range(len(df.columns))]
    returns = []
    for i in range(len(df)):
        if sum(poids) < 0.9999:
            raise ValueError("Sum des poids différent de 1, sum(poids) : ", sum(poids))
        rendement = np.dot(df.iloc[i], poids)
        returns.append(rendement)
    return returns
def BHP(df,stat,end):
    """
        Input : 
            df : dataframe
                type : dataframe
            start : debut de période
                type : datetime
            end : fin de période
                type : datetime
        Output : 
            liste des rendements de la stratégie

        Stratégie : Basé sur https://www.cs.princeton.edu/courses/archive/spring08/cos511/scribe_notes/0428.pdf
            A chaque début de journée, on achete et on revend à la fin.
    """
    poids = np.full(len(df.columns),1/len(df.columns))
    returns = []
    for i in range(len(df)):
        if sum(poids) < 0.9999:
            raise ValueError("Sum des poids différent de 1, sum(poids) : ", sum(poids))
        rendement = np.dot(df.iloc[i], poids)
        returns.append(rendement)
        poids = poids *df.iloc[i].values
        poids = poids/sum(poids)
    return returns