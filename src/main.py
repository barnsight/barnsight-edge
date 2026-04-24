"""Executable entry point for the BarnSight Edge inference worker."""

import signal

from src.inference.worker import InferenceWorker


def main() -> None:
  """Start the edge inference worker."""
  worker = InferenceWorker()
  signal.signal(signal.SIGINT, worker.stop)
  signal.signal(signal.SIGTERM, worker.stop)
  worker.setup()
  worker.run()


if __name__ == "__main__":
  main()
