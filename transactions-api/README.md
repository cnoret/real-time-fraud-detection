---
title: Fraud Transactions API
emoji: 💳
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.45.0
app_file: app.py
pinned: false
license: mit
short_description: Flux de paiements par carte rejoué en temps réel
---

# Fraud Transactions API

Source de transactions du projet [real-time-fraud-detection](https://github.com/cnoret/real-time-fraud-detection).
Elle remplace l'API fournie pendant la formation : le pipeline ne dépend plus d'un service tiers.

L'API rejoue un échantillon fixe de **50 000 transactions** (dont **195 fraudes**, 0,39 %)
tiré du jeu simulé *Sparkov* (`fraudTest.csv`), comme si les paiements arrivaient maintenant.

| Endpoint | Rôle |
|---|---|
| `GET /health` | état du service, volume et taux de fraude du jeu |
| `GET /v1/transactions?n=1&seed=` | `n` transactions (1 à 100) **sans** le label, horodatées `event_time` |
| `GET /v1/labels?trans_num=…` | vérité terrain `is_fraud` (simule le retour des rétrofacturations) |
| `GET /current-transactions` | compatibilité avec l'ancien format (pandas `orient=split`) |

Le label est servi à part, volontairement : il ne peut pas se glisser dans les variables envoyées
au modèle, et il permet de mesurer précision et rappel en production.

**Holdout** : ces 50 000 `trans_num` doivent être exclus de tout réentraînement du modèle,
sinon la mesure de performance serait biaisée.

## Hébergement

Le Space utilise le SDK **Gradio** (gratuit) : Hugging Face exécute `python app.py`, qui lance
l'application FastAPI avec uvicorn sur le port 7860. Gradio n'est pas utilisé dans le code.
Le `Dockerfile` reste disponible pour un lancement en conteneur, en local ou chez un autre hébergeur.

## Lancer en local

```bash
pip install -r requirements.txt
uvicorn app:app --port 7860
# http://localhost:7860/docs
```

Reconstruire le jeu : `python scripts/build_holdout.py` (graine 42, reproductible).

Limite : 60 appels par minute et par client (hébergement gratuit).
