FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# The model lives on the Hugging Face Hub (free downloads), not in git.
ARG MODEL_URL=https://huggingface.co/spaces/cnoret/fraud-detection-api/resolve/main/fraud_model.pkl
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
