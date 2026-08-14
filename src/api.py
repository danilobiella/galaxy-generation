"""FastAPI service exposing the trained diffusion model.

The service commits to DDIM sampling and to a fixed set of step counts. Both
`n_samples` and `steps` are static arguments of the compiled sampler, so any
unseen value costs a ~6s XLA compile and caches a kernel forever. Everything
servable is compiled once at startup instead.
"""

import io
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import jax
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from config import (
    API_MAX_BATCH,
    API_STEPS,
    DDIM_ETA,
    DDIM_STEPS,
    IMAGE_SIZE,
    WEIGHTS_PATH,
)
from generate import load_generator, random_seed
from logs import setup_logging

SAMPLER = "ddim"
IMMUTABLE = "public, max-age=31536000, immutable"
STATIC_DIR = Path(__file__).parent / "static"

logger = setup_logging()

_state: dict = {}
_device_lock = threading.Lock()


def get_generator():
    generator = _state.get("generator")
    if generator is None:
        raise HTTPException(status_code=503, detail="Model still loading.")
    return generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = jax.devices()[0].platform
    logger.info(f"starting up on {device.upper()}")

    start = time.perf_counter()
    logger.info(f"loading weights from {WEIGHTS_PATH} ...")
    generator = load_generator()
    logger.info(f"weights loaded in {time.perf_counter() - start:.2f}s")

    for steps in API_STEPS:
        warm_start = time.perf_counter()
        generator.generate(n_samples=1, seed=0, sampler=SAMPLER, steps=steps)
        logger.info(
            f"compiled sampler steps={steps} in {time.perf_counter() - warm_start:.2f}s"
        )

    _state["generator"] = generator
    logger.info(f"ready in {time.perf_counter() - start:.2f}s total")

    yield

    logger.info("shutting down")
    _state.clear()


app = FastAPI(
    title="Galaxy Generation API",
    description="Diffusion-model galaxy images, sampled on demand.",
    version="0.1.0",
    lifespan=lifespan,
)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def to_png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def validate_steps(steps: int) -> int:
    if steps not in API_STEPS:
        logger.warning(f"rejected steps={steps} (allowed {list(API_STEPS)})")
        raise HTTPException(
            status_code=422,
            detail=f"steps must be one of {list(API_STEPS)}; "
            "other values would require an uncached compile.",
        )
    return steps


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if "generator" in _state else "loading",
        "device": jax.devices()[0].platform,
        "image_size": list(IMAGE_SIZE),
        "sampler": SAMPLER,
        "default_steps": DDIM_STEPS,
        "allowed_steps": list(API_STEPS),
        "max_batch": API_MAX_BATCH,
    }


@app.get(
    "/galaxy",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
def galaxy(
    seed: int | None = Query(None, ge=0, lt=2**31),
    steps: int = Query(DDIM_STEPS),
    eta: float = Query(DDIM_ETA, ge=0.0, le=1.0),
) -> Response:
    """One galaxy as a PNG. Deterministic in `seed` when eta=0."""
    validate_steps(steps)
    generator = get_generator()
    seed = random_seed() if seed is None else seed

    logger.info(f"request  seed={seed} steps={steps} eta={eta}")
    requested = time.perf_counter()

    with _device_lock:
        acquired = time.perf_counter()
        images = generator.generate(
            n_samples=1, seed=seed, sampler=SAMPLER, steps=steps, eta=eta
        )
        sampled = time.perf_counter()

    png = to_png(images[0])
    done = time.perf_counter()

    logger.info(
        f"done     seed={seed} in {done - requested:.2f}s "
        f"(queue {acquired - requested:.2f}s, "
        f"sample {sampled - acquired:.2f}s, "
        f"encode {done - sampled:.2f}s, {len(png) / 1024:.1f} KB)"
    )

    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Cache-Control": IMMUTABLE if eta == 0.0 else "no-store",
            "ETag": f'"{SAMPLER}-{steps}-{eta}-{seed}"',
            "X-Galaxy-Seed": str(seed),
        },
    )


@app.get("/galaxies")
def galaxies(
    n: int = Query(10, ge=1, le=API_MAX_BATCH),
    seed: int | None = Query(None, ge=0, lt=2**31),
    steps: int = Query(DDIM_STEPS),
) -> dict:
    """Seeds for a gallery. The client fetches each image from /galaxy."""
    validate_steps(steps)
    get_generator()
    seed = random_seed() if seed is None else seed

    rng = np.random.default_rng(seed)
    seeds = [int(s) for s in rng.integers(0, 2**31, size=n)]

    logger.info(f"gallery  n={n} base_seed={seed} steps={steps}")

    return {
        "seed": seed,
        "count": n,
        "galaxies": [
            {"seed": s, "url": f"/galaxy?seed={s}&steps={steps}"} for s in seeds
        ],
    }
