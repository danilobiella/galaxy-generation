"""Terminal logging: coloured and timestamp-free interactively, plain in files."""

import logging
import os
import re
import sys

import jax

LOGGER_NAME = "galaxy"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
LEVEL_COLORS = {
    logging.DEBUG: "\033[34m",
    logging.INFO: GREEN,
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}


def use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def green(text: str) -> str:
    return f"{GREEN}{text}{RESET}" if use_color() else text


ANSI = re.compile(r"\033\[[0-9;]*m")


def paint(text: str, code: str, color: bool) -> str:
    return f"{code}{text}{RESET}" if color else text


def pad(text: str, width: int) -> str:
    return text + " " * (width - len(ANSI.sub("", text)))


def format_run_start(n_epochs: int, steps: int, color: bool | None = None) -> str:
    color = use_color() if color is None else color
    return "\n" + paint(f"Training: {n_epochs} epochs × {steps} steps", BOLD, color)


def format_epoch_header(epoch: int, n_epochs: int, color: bool | None = None) -> str:
    color = use_color() if color is None else color
    title = f"── epoch {epoch}/{n_epochs} "
    return paint(title + "─" * (RULE_WIDTH - len(title)), BOLD + CYAN, color)


def format_step_line(
    step: int, steps: int, loss: float, grad_norm: float, lr: float,
    color: bool | None = None,
) -> str:
    color = use_color() if color is None else color
    sep = paint(" | ", DIM, color)
    return (
        paint(f"{'step':<{LABEL_WIDTH}}", BLUE, color)
        + f"{f'{step}/{steps}':<12}"
        + f"loss {paint(f'{loss:.4f}', BOLD, color)}" + sep
        + f"grad {grad_norm:.2e}" + sep
        + f"lr {lr:.2e}"
    )


class ColorFormatter(logging.Formatter):
    def __init__(self, color: bool, timestamps: bool):
        super().__init__(datefmt="%H:%M:%S")
        self.color = color
        self.timestamps = timestamps

    def format(self, record: logging.LogRecord) -> str:
        level = f"{record.levelname:<8}"
        tag = record.name
        parts = []

        if self.color:
            level = f"{LEVEL_COLORS.get(record.levelno, '')}{level}{RESET}"
            tag = f"{CYAN}{tag}{RESET}"

        if self.timestamps:
            stamp = self.formatTime(record, self.datefmt)
            parts.append(f"{DIM}{stamp}{RESET}" if self.color else stamp)

        parts.append(f"{level} {tag}  {record.getMessage()}")
        line = "  ".join(parts)

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


def setup_logging(
    name: str = LOGGER_NAME,
    level: int = logging.INFO,
    color: bool | None = None,
) -> logging.Logger:
    """Attach a stdout handler to `name`, once. Returns the logger."""
    logger = logging.getLogger(name)
    color = use_color() if color is None else color

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter(color=color, timestamps=not color))
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(level)
    return logger


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


def format_scores(scores: dict[str, float]) -> str:
    return (
        f"KID {scores['kid'] * 1e3:.2f} ± {scores['kid_std'] * 1e3:.2f} ×10⁻³ | "
        f"FID {scores['fid']:.2f} | "
        f"density {scores['density']:.3f} | coverage {scores['coverage']:.3f}"
    )


def format_timings(timings: dict[str, float]) -> str:
    return ", ".join(f"{k} {format_duration(v)}" for k, v in timings.items())


def peak_memory() -> str:
    parts = []
    for d in jax.local_devices():
        stats = d.memory_stats() or {}
        if "peak_bytes_in_use" in stats:
            used = stats["peak_bytes_in_use"] / 2**30
            limit = stats.get("bytes_limit", 0) / 2**30
            parts.append(f"{d.id}: {used:.1f}/{limit:.1f} GB")
    return ", ".join(parts) or "n/a"


LABEL_WIDTH = 9
VALUE_WIDTH = 46
RULE_WIDTH = 72


def format_epoch_summary(
    record: dict,
    n_epochs: int,
    wall: float,
    elapsed: float,
    eta: float,
    memory: str,
    color: bool | None = None,
) -> str:
    color = use_color() if color is None else color

    def number(value: float) -> str:
        return paint(f"{value:.4f}", BOLD, color)

    def row(label: str, value: str, seconds: float | None = None, detail: str = "") -> list[str]:
        parts = []
        if seconds is not None:
            parts.append(format_duration(seconds))
        if detail:
            parts.append(f"({detail})")
        timing = "  ".join(parts)
        head = paint(f"{label:<{LABEL_WIDTH}}", BLUE, color)
        if not timing:
            return [head + value]
        timing = paint(timing, DIM, color)
        if len(ANSI.sub("", value)) > VALUE_WIDTH:
            return [head + value, " " * (LABEL_WIDTH + VALUE_WIDTH) + timing]
        return [head + pad(value, VALUE_WIDTH) + timing]

    r = record

    def has(*keys: str) -> bool:
        return all(k in r for k in keys)

    lines = row(
        "train", f"loss {number(r['train_loss'])}", r["time_train"],
        f"data wait {format_duration(r['time_data_wait'])}",
    )
    if has("val_mse"):
        lines += row("val", f"mse  {number(r['val_mse'])}", r.get("time_val"))
    if has("val_mse", "train_probe_mse"):
        lines += row(
            "probe", f"mse  {number(r['train_probe_mse'])}",
            r.get("time_probe"),
        )
    if has("kid", "kid_std", "fid", "density", "coverage"):
        kid, rest = format_scores(r).split(" | ", 1)
        lines += row(
            "quality",
            paint(kid, BOLD, color) + " | " + rest,
            r.get("time_quality"),
            format_timings({
                k.removeprefix("time_quality_"): v
                for k, v in r.items()
                if k.startswith("time_quality_")
            }),
        )
    lines += row(
        "time",
        paint(
            f"epoch {format_duration(wall)} | elapsed {format_duration(elapsed)} | "
            f"ETA {format_duration(eta)} | GPU {memory}",
            DIM, color,
        ),
    )
    return "\n".join(lines)
