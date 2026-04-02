# Contributing to BarnSight Edge

Thank you for your interest in contributing to BarnSight Edge. This document provides guidelines and instructions for contributing to the project.

## Code Style

- **Indentation:** 2 spaces (no tabs)
- **Quotes:** Double quotes for strings
- **Line length:** Keep lines under 100 characters
- **Type hints:** Required for all function signatures
- **Docstrings:** Required for all classes and public methods

## Development Setup

1. Fork and clone the repository
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Run tests to verify your setup:
   ```bash
   uv run pytest tests/ -v
   ```

## Making Changes

1. Create a feature branch from `main`
2. Make your changes following the code style guidelines
3. Add tests for new functionality
4. Ensure all tests pass:
   ```bash
   uv run pytest tests/ -v --cov=src
   ```
5. Commit with a clear, descriptive message
6. Open a pull request

## Pull Request Guidelines

- Describe what the PR does and why
- Reference any related issues
- Include test coverage for new code
- Ensure the PR passes all existing tests
- Keep PRs focused — one feature or fix per PR

## Architecture Notes

- **No SQL:** The project uses in-memory data structures only. Do not introduce SQLite or any database dependency.
- **2-space indent:** All Python files use 2-space indentation. This is enforced by the project's ruff configuration in `pyproject.toml`.
- **Region deduplication:** New detections are matched against tracked regions using IoU (Intersection over Union). Detections in the same region within the cooldown window are suppressed to prevent duplicate image sends.
- **API communication:** All events are sent to the BarnSight API via HTTP POST to `/api/v1/events`. See the `barnsight-api` repository for the API schema and endpoint documentation.
- **Docker-first deployment:** Production runs via Docker Compose with one container per camera. The `Dockerfile` uses a multi-stage build on `python:3.11-slim` to keep the image small for edge hardware.

## Docker Development

The project ships with a `Dockerfile` and `docker-compose.yml` for multi-camera orchestration.

- **Multi-stage build:** Builder stage exports minimal requirements, runtime stage uses `python:3.11-slim`
- **One container per camera:** Each camera service in `docker-compose.yml` gets its own `.env.camN` file
- **Resource limits:** On low-power hosts, add `deploy.resources.limits.cpus` per service in compose
- **Log rotation:** JSON-file driver with 10MB max per container, 3 files retained

To test Docker locally:
```bash
docker compose up -d --build
docker compose logs -f cam-barn-01
docker compose down
```

## Testing

- Tests live in the `tests/` directory
- Use `pytest` as the test framework
- Mock external HTTP calls with `responses` or `unittest.mock`
- Do not require a live camera or API server for tests

## Reporting Issues

When reporting bugs, include:

- Python version
- Hardware (CPU/GPU, RAM)
- Camera source type (RTSP, USB, etc.)
- Steps to reproduce
- Relevant log output (JSON format)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
