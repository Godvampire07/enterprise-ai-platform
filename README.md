# Enterprise AI Platform - Backend

This is a production-grade backend foundation for an Enterprise AI Platform built with FastAPI, PostgreSQL, and Redis.

## Features

- **FastAPI**: High performance, asynchronous API.
- **SQLAlchemy 2.0**: Modern ORM for database interactions.
- **Alembic**: Database migrations.
- **JWT Authentication**: Secure access with access and refresh tokens.
- **Role-Based Access Control (RBAC)**: Fine-grained permissions.
- **Redis Caching**: Efficient token management and session storage.
- **Dockerized**: Fully containerized environment with Docker Compose.
- **Structured Logging**: Request ID tracking and execution time monitoring.
- **Global Exception Handling**: Consistent error responses.

## Prerequisites

- [Docker](https://www.docker.com/get-started) and Docker Compose.
- Python 3.12+ (for local development).

## Quick Start (Docker)

1. **Clone the repository.**
2. **Setup environment variables:**
   ```bash
   cp .env.example .env
   ```
3. **Run the application:**
   ```bash
   docker-compose up --build
   ```
4. **Access Swagger UI:**
   Navigate to [http://localhost:8000/docs](http://localhost:8000/docs).

## Local Development

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the database and redis (using Docker):**
   ```bash
   docker-compose up db redis -d
   ```
4. **Run migrations:**
   ```bash
   alembic upgrade head
   ```
5. **Start the application:**
   ```bash
   uvicorn backend.app.main:app --reload
   ```

## Project Structure

```text
backend/
  app/
    api/         # API routes (v1)
    core/        # Config, security, logging, exceptions
    database/    # Database session and base model
    dependencies/# FastAPI dependencies
    middleware/  # Custom middlewares
    models/      # SQLAlchemy models
    repositories/# Data access layer
    schemas/     # Pydantic validation schemas
    services/    # Business logic layer
    main.py      # Entry point
tests/           # Pytest suites
alembic/         # Migration scripts
```

## Running Tests

```bash
pytest
```
