FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# The model is a versioned asset of a GitHub Release of this repository, not a git file.
# Publishing a new model = new release tag + bump MODEL_VERSION here.
ARG MODEL_VERSION=model-v1.0.0
ARG MODEL_URL=https://github.com/cnoret/real-time-fraud-detection/releases/download/${MODEL_VERSION}/fraud_model.pkl
COPY --chown=user scripts/ scripts/
RUN python scripts/fetch_model.py "$MODEL_URL" fraud_model.pkl > sklearn_version.txt

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "scikit-learn==$(cat sklearn_version.txt)"

# Fail the build, not the first request, if the model cannot be loaded
RUN python -c "import joblib; m = joblib.load('fraud_model.pkl'); print('Model OK:', type(m).__name__, list(m.named_steps))"

COPY --chown=user app.py .

# Hosts such as Render inject $PORT; default to 7860 elsewhere
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
