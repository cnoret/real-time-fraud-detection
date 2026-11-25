# Fraud Detection Model API

API FastAPI qui sert le modèle de détection de fraude (pipeline scikit-learn entraîné dans
`EDA_training.ipynb`). Le DAG Airflow l'appelle pour chaque transaction.

| Endpoint | Rôle |
|---|---|
| `GET /health` | état du service et colonnes attendues par le modèle |
| `POST /predict` | probabilité de fraude, décision et niveau de risque pour une transaction |
| `POST /predict/batch` | la même chose pour une liste de transactions |
| `GET /thresholds` | seuils de décision disponibles (`balanced` = 0,1 par défaut) |

Exemple :

```bash
curl -X POST https://fraud-model-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"amt": 950.0, "category": "shopping_net", ...}, "threshold": "balanced"}'
```

## Modèle

Le fichier `fraud_model.pkl` (environ 32 Mo) n'est pas versionné dans git : il est téléchargé
depuis le Hugging Face Hub pendant la construction de l'image (argument `MODEL_URL` du
`Dockerfile`). `scripts/fetch_model.py` lit dans le fichier la version de scikit-learn qui l'a
enregistré, et l'image installe exactement cette version : un modèle scikit-learn ne se recharge
de façon fiable qu'avec la même version. La construction échoue si le modèle ne se charge pas.

## Hébergement

Déployée sur **Render** (offre gratuite) avec l'API de transactions, via `render.yaml` à la
racine du dépôt. Le service se met en veille après 15 minutes sans trafic et se réveille au
premier appel (environ une minute).

En local :

```bash
docker build -t fraud-model-api model-api
docker run -p 7860:7860 fraud-model-api   # http://localhost:7860/docs
```
