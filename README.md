# SRS Analyzer

This project is a FastAPI-based system that takes a Software Requirements Specification (SRS) document as input, analyzes its content, and generates an AI-powered FastAPI project following best practices in software engineering.

## Features
- Analyze SRS documents (text or image).
- Generate project folder structure and code.
- Set up and integrate a PostgreSQL database.
- Automated debugging and testing.
- Seamless deployment with Docker.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```
3. Access the API at `http://127.0.0.1:8000`.

## Deployment
To deploy using Docker:
1. Build the Docker image:
   ```bash
   docker build -t srs-analyzer .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 srs-analyzer
   ```