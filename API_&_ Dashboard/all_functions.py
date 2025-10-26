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


def preprocess_df(df, id_cols=['TransactionID', 'TransactionDT']):
    """
    Prépare le DataFrame pour le modèle :
    - Encode les colonnes catégorielles en int16/int32
    - Crée les colonnes D1n-D15n
    - Applique Min-Max scaling aux colonnes numériques
    - Exclut les colonnes d'identifiant

    Parameters:
    - df : pd.DataFrame
    - cat_cols : list de colonnes catégorielles
    - id_cols : list de colonnes à exclure du scaling

    Returns:
    - df preprocessé
    """
    df = df.copy()
    
    cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'DeviceType', 'DeviceInfo', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']
    
    # ---- Label encode categorical columns ----
    for col in df.columns:
        if col in cat_cols:
            encoded, _ = pd.factorize(df[col], sort=True)
            if encoded.max() > 32000:
                df[col] = encoded.astype('int32')
            else:
                df[col] = encoded.astype('int16')
    
    # ---- Créer les colonnes D1n à D15n ----
    new_cols = {}
    for i in range(1, 16):
        orig_col = 'D' + str(i)
        if orig_col in df.columns and 'TransactionDT' in df.columns:
            new_cols[orig_col + 'n'] = df[orig_col] - df['TransactionDT'] / np.float32(24*60*60)
    
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    # ---- Scaling des colonnes numériques ----
    rem_cols = id_cols.copy()
    cols_to_scale = [col for col in df.columns if col not in rem_cols + cat_cols]
    
    for col in cols_to_scale:
        scaled = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        scaled.fillna(-1, inplace=True)
        df[col] = scaled
    
    return df



def set_column_types(df):
    """
    Transforme les colonnes catégorielles et numériques du DataFrame df
    en types appropriés (str pour catégorielles, float64 pour numériques),
    sans tenir compte de la colonne target 'isFraud' si elle n'existe pas.
    Évite les SettingWithCopyWarning et les incompatibilités de dtype.
    """
    df = df.copy()  # créer une copie propre

    # Colonnes catégorielles
    cat_cols = (['ProductCD'] +
                ['card%d' % i for i in range(1, 7)] +
                ['addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'] +
                ['M%d' % i for i in range(1, 10)] +
                ['DeviceType', 'DeviceInfo'] +
                ['id_%d' % i for i in range(12, 39)])
    
    # Garder seulement celles qui existent
    cat_cols_existing = [c for c in cat_cols if c in df.columns]

    # Conversion sécurisée en string
    for col in cat_cols_existing:
        df[col] = df[col].astype(str)

    # Colonnes d'identifiant
    id_cols = ['TransactionID', 'TransactionDT']
    id_cols_existing = [c for c in id_cols if c in df.columns]

    # Colonne target
    target = 'isFraud'
    target_existing = [target] if target in df.columns else []

    # Colonnes numériques
    num_cols = [col for col in df.columns if col not in cat_cols_existing + id_cols_existing + target_existing]
    
    # Conversion sécurisée en float64
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # force les valeurs invalides en NaN
        df[col] = df[col].astype('float64')
    
    return df


def filter_columns(df):
    columns_to_keep = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V1', 'V3', 'V4', 'V6', 'V8', 'V11', 'V13', 'V14', 'V17', 'V20', 'V23', 'V26', 'V27', 'V30', 'V36', 'V37', 'V40', 'V41', 'V44', 'V47', 'V48', 'V54', 'V55', 'V56', 'V59', 'V62', 'V65', 'V67', 'V68', 'V70', 'V76', 'V78', 'V80', 'V82', 'V86', 'V88', 'V89', 'V91', 'V96', 'V98', 'V99', 'V107', 'V108', 'V109', 'V111', 'V114', 'V115', 'V117', 'V120', 'V121', 'V123', 'V124', 'V127', 'V129', 'V130', 'V131', 'V172', 'V173', 'V176', 'V178', 'V198', 'V199', 'V201', 'V203', 'V207', 'V209', 'V210', 'V218', 'V220', 'V221', 'V223', 'V226', 'V229', 'V234', 'V238', 'V240', 'V241', 'V250', 'V258', 'V260', 'V264', 'V271', 'V277', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V289', 'V291', 'V294', 'V296', 'V301', 'V303', 'V305', 'V307', 'V309', 'V310', 'V312', 'V314', 'V320', 'id_01', 'id_02', 'id_05', 'id_06', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
    
    return df[columns_to_keep]