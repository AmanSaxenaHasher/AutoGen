# SRS Analyzer

This project is a FastAPI-based system that takes a Software Requirements Specification (SRS) document as input, analyzes its content, and generates an AI-powered FastAPI project following best practices in software engineering.

## Features
- Analyze SRS documents (text or image).
- Generate project folder structure and code.
- Automated debugging and testing.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```
3. Access the API at `http://127.0.0.1:8080`.