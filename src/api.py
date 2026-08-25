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
    CLASS_NAMES,
    DDIM_ETA,
    DDIM_STEPS,
    GUIDANCE_SCALE,
    IMAGE_SIZE,
    NUM_CLASSES,
    WEIGHTS_PATH,
)
from generate import load_generator, random_seed
from logs import setup_logging

SAMPLER = "ddim"
MAX_GUIDANCE = 8.0
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


def resolve_class(generator, class_id: int | None, seed: int) -> int | None:
    """Turn "any" into a concrete class, so the answer stays a function of `seed`.

    Leaving it unresolved would make the ETag lie: two requests with the same
    seed could return different images.
    """
    if not generator.conditional:
        if class_id is not None:
            raise HTTPException(
                status_code=422,
                detail="This checkpoint is unconditional; `class` is not supported.",
            )
        return None

    if class_id is not None:
        return class_id

    return int(np.random.default_rng(seed).integers(NUM_CLASSES))


@app.get("/health")
def health() -> dict:
    generator = _state.get("generator")
    return {
        "status": "ok" if generator is not None else "loading",
        "device": jax.devices()[0].platform,
        "image_size": list(IMAGE_SIZE),
        "sampler": SAMPLER,
        "default_steps": DDIM_STEPS,
        "allowed_steps": list(API_STEPS),
        "max_batch": API_MAX_BATCH,
        "conditional": bool(generator is not None and generator.conditional),
        "class_names": list(CLASS_NAMES),
        "default_guidance": GUIDANCE_SCALE,
        "max_guidance": MAX_GUIDANCE,
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
    class_id: int | None = Query(None, alias="class", ge=0, lt=NUM_CLASSES),
    guidance: float = Query(GUIDANCE_SCALE, ge=0.0, le=MAX_GUIDANCE),
) -> Response:
    """One galaxy as a PNG. Deterministic in `seed` when eta=0.

    `class` selects a morphology, or picks one from the seed when omitted.
    `guidance` is the classifier-free scale: 0 ignores the class entirely and
    reproduces the unconditional model, 1 is plain conditional sampling, higher
    trades diversity for class fidelity.
    """
    validate_steps(steps)
    generator = get_generator()
    seed = random_seed() if seed is None else seed
    class_id = resolve_class(generator, class_id, seed)

    logger.info(
        f"request  seed={seed} steps={steps} eta={eta} "
        f"class={class_id} guidance={guidance}"
    )
    requested = time.perf_counter()

    with _device_lock:
        acquired = time.perf_counter()
        images = generator.generate(
            n_samples=1,
            seed=seed,
            sampler=SAMPLER,
            steps=steps,
            eta=eta,
            labels=class_id,
            guidance=guidance,
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

    headers = {
        "Cache-Control": IMMUTABLE if eta == 0.0 else "no-store",
        "ETag": f'"{SAMPLER}-{steps}-{eta}-{seed}-{class_id}-{guidance}"',
        "X-Galaxy-Seed": str(seed),
    }
    if class_id is not None:
        headers["X-Galaxy-Class"] = str(class_id)
        headers["X-Galaxy-Guidance"] = str(guidance)

    return Response(content=png, media_type="image/png", headers=headers)


@app.get("/galaxies")
def galaxies(
    n: int = Query(10, ge=1, le=API_MAX_BATCH),
    seed: int | None = Query(None, ge=0, lt=2**31),
    steps: int = Query(DDIM_STEPS),
    class_id: int | None = Query(None, alias="class", ge=0, lt=NUM_CLASSES),
    guidance: float = Query(GUIDANCE_SCALE, ge=0.0, le=MAX_GUIDANCE),
) -> dict:
    """Seeds for a gallery. The client fetches each image from /galaxy.

    Every returned URL names a concrete class, so an omitted `class` gives a
    mixed batch while each individual image stays cacheable.
    """
    validate_steps(steps)
    generator = get_generator()
    seed = random_seed() if seed is None else seed

    rng = np.random.default_rng(seed)
    seeds = [int(s) for s in rng.integers(0, 2**31, size=n)]

    if not generator.conditional:
        resolve_class(generator, class_id, seed)
        classes: list[int | None] = [None] * n
    elif class_id is None:
        classes = [int(c) for c in rng.integers(NUM_CLASSES, size=n)]
    else:
        classes = [class_id] * n

    if not generator.conditional:
        described = "none"
    else:
        described = "mixed" if class_id is None else str(class_id)

    logger.info(
        f"gallery  n={n} base_seed={seed} steps={steps} "
        f"class={described} guidance={guidance}"
    )

    entries = []
    for image_seed, image_class in zip(seeds, classes):
        url = f"/galaxy?seed={image_seed}&steps={steps}"
        entry = {"seed": image_seed, "url": url}
        if image_class is not None:
            entry["class_id"] = image_class
            entry["class_name"] = CLASS_NAMES[image_class]
            entry["guidance"] = guidance
            entry["url"] = f"{url}&class={image_class}&guidance={guidance}"
        entries.append(entry)

    return {"seed": seed, "count": n, "galaxies": entries}
