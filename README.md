# Détection de Fraude - Projet IEEE-CIS Kaggle

Ce projet vise à développer un modèle de machine learning capable d’identifier des transactions frauduleuses à partir de données issues du concours Kaggle IEEE-CIS Fraud Detection.  
Il simule les besoins réels d’une banque souhaitant réduire les pertes liées à la fraude.


## Objectifs

- Analyser et comprendre les données de transactions.
- Créer et comparer différents modèles de détection de fraude.
- Évaluer la performance selon des métriques adaptées (AUC, précision, rappel, F1).
- Démontrer un pipeline reproductible pour une utilisation en contexte bancaire.


## Jeu de données

Les données proviennent du concours [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection).  
Elles contiennent :
- Données transactionnelles : montants, horodatages, device, etc.
- Données client anonymisées : identifiants, caractéristiques dérivées.
- Variable cible : `isFraud` (1 = fraude, 0 = transaction normale).


## Organisation des notebooks

- `01_data_exploration.ipynb` → Analyse exploratoire (EDA)
- `02_correlation_analysis
.ipynb` → Nettoyage & création de features
- `03_modeling_baseline.ipynb` → Modèle de base



## Installation

Clonez le projet :
```bash
git clone https://github.com/tonpseudo/ieee-fraud-detection.git
cd ieee-fraud-detection

# Créez un environnement virtuel et activez-le
windows:
python -m venv venv
venv\Scripts\activate

MacOS/Linux:
python3 -m venv venv
source venv/bin/activate

Installez les dépendances :
pip install -r requirements.txt

Extraire les donnees:
python extract_data.py

Lancez l'api et le dashboard:
python API_&_ Dashboard/start_services.py
```

## Utilisation

Exécutez les notebooks dans l’ordre pour reproduire les résultats :  
1. Exploration des données  
2. Analyse de corrélation
3. Modélisation, Evaluation et Validation 


---



## Méthodologie & Résultats

### Objectif du projet
L’objectif de ce projet est de construire un modèle de **détection de fraude sur les transactions bancaires** à partir du jeu de données du concours **IEEE-CIS Fraud Detection**.  
Ce dataset est **hautement déséquilibré** (≈ 96.5 % de transactions normales contre 3.5 % de transactions frauduleuses), ce qui rend la tâche de classification particulièrement complexe.

---

### Méthodologie

#### Préparation des données
- Nettoyage et suppression des colonnes inutiles.
- Transformation temporelle :
  ```python
  for i in range(1, 16):
      df["D" + str(i) + "n"] = df["D" + str(i)] - df["TransactionDT"] / (24*60*60)
    ```
- Encodage des variables catégorielles par Label Encoding.
- Mise à l’échelle (Min-Max Scaling) des variables numériques.
- Gestion du déséquilibre des classes grâce à :
- scale_pos_weight dans XGBoost et CatBoost,
- class_weight dans LightGBM.


#### Entraînement des modèles
Trois modèles ont été testés et comparés :

| Modèle       | Bibliothèque | Méthode d’équilibrage | Particularité                                             |
| ------------ | ------------ | --------------------- | --------------------------------------------------------- |
| **XGBoost**  | `xgboost`    | `scale_pos_weight`    | Rapide, robuste sur les features bruitées                 |
| **CatBoost** | `catboost`   | `scale_pos_weight`    | Excellente gestion automatique des features catégorielles |
| **LightGBM** | `lightgbm`   | `class_weight`        | Très performant sur grands volumes de données             |

Tous les modèles ont été évalués sur la même partition train/test à l’aide de la ROC-AUC comme métrique principale.

#### Évaluation des performances

| Modèle          |  ROC-AUC  | Précision fraude | Fausse alerte (FPR) | Vraie détection (TPR) |
| :-------------- | :-------: | :--------------: | :-----------------: | :-------------------: |
| 🥇 **XGBoost**  | **0.949** |       Bonne      |        Faible       |         Élevée        |
| 🥈 **CatBoost** |   0.937   |    Très stable   |       Moyenne       |         Élevée        |
| 🥉 **LightGBM** |   0.921   |     Correcte     |  Un peu plus élevée |        Moyenne        |


#### Matrices de confusion (résumé)

Matrices de confusion (résumé)


---
### Sélection du modèle final

Le modèle XGBoost a été retenu comme modèle final pour plusieurs raisons :

- Meilleur score ROC-AUC (≈ 0.95).
- Bon compromis entre sensibilité (détection de fraudes) et spécificité (peu de faux positifs).
- Bonne vitesse d’inférence, idéale pour un déploiement en API temps réel.

Le modèle final a été sauvegardé en format .joblib :
```python
joblib.dump(best_model, "best_model.joblib")
```

.


## ⚠️ Limites et pistes d'amélioration

- Données anonymisées → certaines variables difficiles à interpréter.
- Pas d’accès aux données en temps réel.
- Possibilité d’améliorer avec des modèles deep learning (Transformers, TabNet).
- Besoin d’un monitoring pour détecter le drift des données.


## Auteurs

Projet réalisé par Mackenson JOSEPH et Abdielson LYVERT dans le cadre d’un bootcamp de Data Science.  
Basé sur le dataset IEEE-CIS Kaggle.