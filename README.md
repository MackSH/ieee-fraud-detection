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
- `02_feature_engineering.ipynb` → Nettoyage & création de features
- `03_modeling_baseline.ipynb` → Modèle de base
- `04_modeling_advanced.ipynb` → Modèles avancés & tuning
- `05_evaluation_and_validation.ipynb` → Comparaison & validation
- `06_deployment_preparation.ipynb` → Export & pipeline
- `07_reporting.ipynb` → Résumé exécutif & visualisations


## Installation

Clonez le projet :
```bash
git clone https://github.com/tonpseudo/ieee-fraud-detection.git

cd ieee-fraud-detection

pip install -r requirements.txt
```



---


## Utilisation

Exécutez les notebooks dans l’ordre pour reproduire les résultats :  
1. Exploration des données  
2. Feature engineering  
3. Modélisation  
4. Évaluation  

Vous pouvez également lancer le pipeline d’entraînement complet : python train_pipeline.py


## Résultats

- Modèle retenu : LightGBM + features métiers
- AUC : 0.922
- Précision : 89%
- Rappel : 76%

👉 Explicabilité : analyse des features importantes avec SHAP.


## ⚠️ Limites et pistes d'amélioration

- Données anonymisées → certaines variables difficiles à interpréter.
- Pas d’accès aux données en temps réel.
- Possibilité d’améliorer avec des modèles deep learning (Transformers, TabNet).
- Besoin d’un monitoring pour détecter le drift des données.


## Auteurs

Projet réalisé par Mackenson JOSEPH et Abdielson LYVERT dans le cadre d’un bootcamp de Data Science.  
Basé sur le dataset IEEE-CIS Kaggle.