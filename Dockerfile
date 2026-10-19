FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    JAX_PLATFORMS=cpu \
    PORT=7860

WORKDIR /app

COPY --chown=user requirements-inference.txt .
RUN pip install --no-cache-dir --user -r requirements-inference.txt

COPY --chown=user src/ /app/src/
COPY --chown=user model_checkpoints/galaxy_unet_ema.safetensors /app/model_checkpoints/galaxy_unet_ema.safetensors

EXPOSE 7860

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
