"""JSON-structured logging configuration for BarnSight Edge.

Produces machine-parseable log output with UTC timestamps,
suitable for log aggregation and monitoring systems.
"""

import json
import logging
from datetime import datetime, timezone
from logging.config import dictConfig


class JsonFormatter(logging.Formatter):
  """Format log records as single-line JSON objects."""

  def format(self, record):
    log_record = {
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "level": record.levelname,
      "logger": record.name,
      "module": record.module,
      "line": record.lineno,
      "message": record.getMessage(),
    }
    if record.exc_info:
      log_record["exception"] = self.formatException(record.exc_info)
    return json.dumps(log_record)


log_config = {
  "version": 1,
  "disable_existing_loggers": False,
  "formatters": {
    "json": {
      "()": JsonFormatter,
    },
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": "DEBUG",
      "formatter": "json",
      "stream": "ext://sys.stdout",
    },
  },
  "loggers": {
    "app": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
  },
  "root": {"handlers": ["console"], "level": "DEBUG"},
}

dictConfig(log_config)
logger = logging.getLogger("app")
