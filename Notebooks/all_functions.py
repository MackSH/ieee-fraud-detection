import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def top_missing_cols(df,n=10,thresh=80):
    """
    Renvoie les colonnes manquantes dans le dataframe avec des valeurs manquantes de pourcentage > threesh si n = Aucun. Le dataframe complet sera renvoyé avec des valeurs manquantes de pourcentage > threesh.
    """
    
    dff = (df.isnull().sum()/df.shape[0])*100
    dff = dff.reset_index()
    dff.columns = ['col','missing_percent']
    dff = dff.sort_values(by=['missing_percent'],ascending=False).reset_index(drop=True)
    print(f'Il y a {df.isnull().any().sum()} colonnes dans ce jeu de données contenant des valeurs manquantes.')
    print(f'Il y a {dff[dff["missing_percent"] > thresh].shape[0]} colonnes dont le pourcentage de valeurs manquantes est supérieur à {thresh}%.')

    if n:
        return dff.head(n)
    else:
        return dff
    

def correlation_analysis(df, cols,title='Coorelation Analysis',size=(12,12)):
    cols = sorted(cols)
    fig,axes = plt.subplots(1,1,figsize=size)
    df_corr = df[cols].corr()
    sns.heatmap(df_corr,annot=True,cmap='RdBu_r')
    axes.title.set_text(title)
    plt.show()


def reduce_groups(df, grps):
    '''
    déterminer la colonne qui a le plus de valeurs uniques parmi un groupe d'attributs
    '''
    use = []
    for col in grps:
        max_unique = 0
        max_index = 0
        for i,c in enumerate(col):
            n = df[c].nunique()
            if n > max_unique:
                max_unique = n
                max_index = i
        use.append(col[max_index])
    return use