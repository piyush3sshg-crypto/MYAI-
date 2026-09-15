"""Centralized logging — before this, the project only ever used
print() statements, so there was no way to distinguish info/warnings/
errors, filter by severity, or keep a persistent record of what
happened during a run. This wires up Python's standard `logging`
module (no external dependency) for that.
"""
import logging
import os


_CONFIGURED = False


def get_logger(name="myai", log_dir="logs", level=logging.INFO, also_console=True):
    """Get a configured logger. Safe to call repeatedly — configuration
    only happens once per process.

    Log file: logs/myai.log — plain text, one line per event, rotated
    by size (1MB, keep last 3) so it can't grow unbounded on a phone
    with limited storage.
    """
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        try:
            os.makedirs(log_dir, exist_ok=True)
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                os.path.join(log_dir, "myai.log"),
                maxBytes=1_000_000, backupCount=3, encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            # e.g. read-only filesystem — fall back to console-only
            pass

        if also_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        _CONFIGURED = True

    return logger


def log_prediction(logger, component, input_text, output, confidence=None, duration_ms=None):
    """Standard structured log line for any model prediction — the
    kind of record you'd want for monitoring a real deployment."""
    parts = [f"component={component}", f"input_len={len(input_text)}", f"output={output}"]
    if confidence is not None:
        parts.append(f"confidence={confidence:.4f}")
    if duration_ms is not None:
        parts.append(f"duration_ms={duration_ms:.1f}")
    logger.info("prediction " + " ".join(parts))


def log_error(logger, component, error, context=None):
    context_str = f" context={context}" if context else ""
    logger.error(f"component={component} error={type(error).__name__}: {error}{context_str}")
