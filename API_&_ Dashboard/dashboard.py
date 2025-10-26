import streamlit as st
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config Streamlit ---
st.set_page_config(page_title="🧠 Fraud Detection Dashboard", layout="wide")

st.title("🧠 Fraud Detection Dashboard")

# --- Upload fichiers ---
transaction_file = st.file_uploader("📂 Transaction CSV", type=["csv"])
identity_file = st.file_uploader("📂 Identity CSV (optionnel)", type=["csv"])
process_btn = st.button("🚀 Lancer la prédiction")

# Stocker les prédictions dans la session
if "df_pred" not in st.session_state:
    st.session_state.df_pred = None

if process_btn and transaction_file:
    with st.spinner("⚡ Prédiction en cours..."):
        files = {"transaction_file": transaction_file}
        if identity_file:
            files["identity_file"] = identity_file
        response = requests.post("http://localhost:3003/predict_csv/", files=files)
        if response.status_code == 200:
            st.session_state.df_pred = pd.read_csv(io.BytesIO(response.content))
            st.success("✅ Prédiction terminée !")
        else:
            st.error(f"Erreur API: {response.status_code}")

# --- Si données disponibles ---
if st.session_state.df_pred is not None:
    df_pred = st.session_state.df_pred.copy()

    # --- Paramètres généraux pour les graphiques ---
    st.sidebar.header("🎨 Paramètres d'affichage")
    width = st.sidebar.slider("📏 Largeur du graphique", 6, 20, 12)
    height = st.sidebar.slider("📐 Hauteur du graphique", 4, 12, 5)

    # --- Recherche & modification de transaction ---
    st.subheader("🔍 Rechercher et modifier une transaction")

    col_search, col_edit = st.columns([2, 1])
    with col_search:
        tx_id = st.text_input("Entrez l'ID de la transaction à rechercher (TransactionID):")

    if tx_id:
        try:
            tx_id = int(tx_id)
            result = df_pred[df_pred['TransactionID'] == tx_id]

            if not result.empty:
                st.write("### Transaction trouvée :")
                st.dataframe(result)

                new_label = st.radio("Changer le label isFraud :", [0, 1],
                                     index=int(result['isFraud'].iloc[0]),
                                     horizontal=True)

                if st.button("💾 Mettre à jour"):
                    df_pred.loc[df_pred['TransactionID'] == tx_id, 'isFraud'] = new_label
                    st.session_state.df_pred = df_pred
                    st.success("✅ Transaction mise à jour avec succès !")
                    st.rerun()
            else:
                st.warning("⚠️ Aucune transaction trouvée avec cet ID.")
        except ValueError:
            st.error("❌ L'ID doit être un nombre entier.")

    # --- Résumé global ---
    st.header("📊 Résumé global")
    total_tx = len(df_pred)
    total_fraud = df_pred['isFraud'].sum()
    perc_fraud = total_fraud / total_tx * 100
    total_amount = df_pred['TransactionAmt'].sum()
    total_fraud_amount = df_pred.loc[df_pred['isFraud'] == 1, 'TransactionAmt'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total transactions", total_tx)
    col2.metric("Transactions frauduleuses", total_fraud, f"{perc_fraud:.2f}%")
    col3.metric("Montant total", f"${total_amount:,.2f}")
    col4.metric("Montant fraude estimé", f"${total_fraud_amount:,.2f}")

    # --- Ligne 1 : Distribution & Top transactions ---
    colA, colB = st.columns(2)

    with colA:
        st.subheader("🎯 Distribution isFraud")
        fig, ax = plt.subplots(figsize=(width, height))
        sns.countplot(x='isFraud', data=df_pred, ax=ax)
        total = len(df_pred)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 3,
                    f"{height*100/total:.2f}%", ha='center')
        ax.set_title("Répartition des transactions")
        st.pyplot(fig)

    with colB:
        st.subheader("⚠️ Top 10 transactions à risque")
        top_risk = df_pred.sort_values("isFraud", ascending=False).head(10)
        st.dataframe(top_risk[['TransactionID', 'TransactionAmt', 'card1', 'ProductCD', 'isFraud']])

    # --- Ligne 2 : Fraude par jour de semaine & Histogramme ---
    df_pred['dayofweek'] = (df_pred['TransactionDT']//(60*60*24)-1)%7

    tmp = df_pred[['isFraud', 'dayofweek']].groupby('dayofweek').mean().reset_index() \
        .rename(columns={'isFraud': 'Percentage fraud transactions'})
    tmp_count = df_pred[['TransactionID', 'dayofweek']].groupby('dayofweek').count().reset_index() \
        .rename(columns={'TransactionID': 'Number of transactions'})
    tmp = tmp.merge(tmp_count, on='dayofweek')

    colC, colD = st.columns(2)

    with colC:
        st.subheader("📅 Fraudes vs Jour de la semaine")
        fig2, ax2 = plt.subplots(figsize=(width, height))
        sns.lineplot(x='dayofweek', y='Percentage fraud transactions', data=tmp, color='r', ax=ax2)
        ax3 = ax2.twinx()
        sns.barplot(x='dayofweek', y='Number of transactions', data=tmp, palette='summer', ax=ax3)
        ax2.set_title("Fraude par jour de la semaine")
        st.pyplot(fig2)

    with colD:
        st.subheader("📈 Distribution des prédictions de fraude")
        fig3, ax3 = plt.subplots(figsize=(width, height))
        sns.histplot(df_pred['isFraud'], bins=2, ax=ax3, color='green')
        ax3.set_title("Histogramme des prédictions de fraude")
        st.pyplot(fig3)

    # --- Ligne 3 : Montant vs Fraude & Autre graphique exemple ---
    colE, colF = st.columns(2)

    with colE:
        st.subheader("💵 Montant moyen par statut de fraude")
        avg_amt = df_pred.groupby('isFraud')['TransactionAmt'].mean().reset_index()
        fig4, ax4 = plt.subplots(figsize=(width, height))
        sns.barplot(x='isFraud', y='TransactionAmt', data=avg_amt, palette='coolwarm', ax=ax4)
        ax4.set_title("Montant moyen des transactions")
        st.pyplot(fig4)

    with colF:
        st.subheader("🧩 Exemple de graphique libre")
        fig5, ax5 = plt.subplots(figsize=(width, height))
        sns.boxplot(x='isFraud', y='TransactionAmt', data=df_pred, ax=ax5, palette='mako')
        ax5.set_title("Distribution des montants par statut")
        st.pyplot(fig5)

    # --- Détails ---
    st.header("🗂️ Détails des transactions")
    st.dataframe(df_pred)
