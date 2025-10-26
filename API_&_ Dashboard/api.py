# from fastapi import FastAPI, UploadFile, File
# import pandas as pd
# import numpy as np
# import joblib
# import io

# from all_functions import filter_columns
# from all_functions import set_column_types
# from all_functions import preprocess_df

# app = FastAPI(title="API de prédiction")

# # Chargement du modèle
# model = joblib.load("best_model.joblib")


# # Colonnes utiles (adapter selon ton dataset)


# @app.post("/predict_csv/")
# async def predict_csv(transaction_file: UploadFile = File(...),
#                       identity_file: UploadFile = File(None)):
#     # Lecture du CSV transaction
#     trans = pd.read_csv(io.BytesIO(await transaction_file.read()))
    
#     # Lecture CSV identity si fourni
#     if identity_file:
#         identity = pd.read_csv(io.BytesIO(await identity_file.read()))
#         df = trans.merge(identity, on="TransactionID", how="left")
#     else:
#         df = trans.copy()
    
    
#     df = filter_columns(df)
#     df = set_column_types(df)
#     df = preprocess_df(df)
    
#     # Prédiction
#     df["label"] = model['model'].predict(df)
    
#     # Retour CSV
#     output = io.StringIO()
#     df.to_csv(output, index=False)
#     output.seek(0)
    
#     return {"file": output.getvalue()}


from fastapi import FastAPI, Response, UploadFile, File
import pandas as pd
import joblib
import io
import os

from all_functions import filter_columns, set_column_types, preprocess_df

app = FastAPI(title="API de prédiction")

# Charger le modèle
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best_model.joblib")
model = joblib.load(model_path)

@app.post("/predict_csv/")
async def predict_csv(
    transaction_file: UploadFile = File(...),
    identity_file: UploadFile = File(None)
):
    # Lecture CSV transaction
    trans = pd.read_csv(io.BytesIO(await transaction_file.read()))

    # Lecture CSV identity si fourni
    if identity_file:
        identity = pd.read_csv(io.BytesIO(await identity_file.read()))
        id_cols = [col for col in identity.columns if col[:2] == 'id']
        rename_cols = {i: 'id_' + str(i[-2:]) for i in id_cols}
        identity = identity.rename(columns=rename_cols)
        df = trans.merge(identity, on="TransactionID", how="left")
    else:
        df = trans

    # Prétraitement
    df = filter_columns(df)
    df = set_column_types(df)
    df = preprocess_df(df)

    # Prédiction
    if 'model' in model:
        preds = model['model'].predict(df)
    else:
        preds = model.predict(df)
    df["isFraud"] = preds

    # Retourne tout le CSV d'un coup
    # return df.to_csv(index=False)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    return Response(content=csv_bytes, media_type="text/csv")
