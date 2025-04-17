import os
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import traceback

from docx import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq


# Initialize Groq LLM
def initialize_groq_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")
    return ChatGroq(api_key=api_key, model="llama-3-70b-8192")


# Analyze SRS document
def analyze_srs_document(srs_document_bytes):
    try:
        doc = Document(BytesIO(srs_document_bytes))
        content = "\n".join([
            paragraph.text.strip().replace("\t", " ")
            for paragraph in doc.paragraphs if paragraph.text.strip()
        ])
    except Exception as e:
        raise ValueError(f"Error reading .docx file: {str(e)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = text_splitter.split_text(content)

    llm = initialize_groq_llm()

    formatted_results = {
        "api_endpoints": [],
        "backend_logic": [],
        "database_schema": [],
        "authentication_requirements": []
    }

    section_prompts = {
        "api_endpoints": "Extract all API endpoints and their parameters. Include the HTTP method, path, and purpose.",
        "backend_logic": "Extract all backend logic. Include business rules and system behavior.",
        "database_schema": "Extract all database schema elements. Include table names, columns, types, and relationships.",
        "authentication_requirements": "Extract authentication and authorization requirements. Include roles, methods, and permissions."
    }

    for section, instructions in section_prompts.items():
        prompt_template = PromptTemplate(
            input_variables=["document"],
            template=f"You are an expert software engineer. {instructions}\n\nDocument:\n{{document}}"
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        section_results = [chain.run(document=chunk) for chunk in chunks]

        formatted_results[section] = [
            result.strip().replace("\n", " ").replace("\t", " ")
            for result in section_results if result.strip()
        ]

    return formatted_results


# Generate Code
def generate_code(requirements):
    print("Generating code...")
    project_root = Path("c:/study/autogen/generated_project")
    if project_root.exists():
        shutil.rmtree(project_root)
    project_root.mkdir(parents=True)

    folders = [
        "app/api/routes", "app/models", "app/services",
        "app/core", "tests"
    ]
    for folder in folders:
        (project_root / folder).mkdir(parents=True, exist_ok=True)

    llm = initialize_groq_llm()

    prompts = {
        "app/main.py": "Generate the main entry point (main.py) for a FastAPI application.",
        "app/api/routes/__init__.py": "Generate FastAPI routes based on typical extracted API endpoints.",
        "app/models/__init__.py": "Generate Pydantic models for database schema and request/response.",
        "app/services/__init__.py": "Generate service layer functions using FastAPI conventions.",
        "app/core/config.py": "Generate configuration (settings, DB URI) for a FastAPI project."
    }

    for path_str, message in prompts.items():
        full_path = project_root / path_str
        response = llm.invoke(message)
        full_path.write_text(response.strip())

    print("Code generation completed.")


# Setup Environment
def setup_environment():
    print("Setting up virtual environment...")
    project_root = Path("c:/study/autogen/generated_project")
    venv_path = project_root / "venv"

    try:
        subprocess.run(["python", "-m", "venv", str(venv_path)], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Virtual environment creation failed: {e}")

    pip_path = venv_path / "Scripts" / "pip"
    requirements_file = Path("c:/study/autogen/srs-analyzer/requirements.txt")

    try:
        if requirements_file.exists():
            subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
        else:
            subprocess.run([str(pip_path), "install", "fastapi", "uvicorn", "psycopg2", "alembic"], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Dependency installation failed: {e}")

    print("Environment setup complete.")


# Setup Database
def setup_database():
    print("Setting up database (PostgreSQL + Alembic)...")
    project_root = Path("c:/study/autogen/generated_project")
    alembic_ini = project_root / "alembic.ini"
    migrations_dir = project_root / "migrations"
    migrations_dir.mkdir(exist_ok=True)

    alembic_ini.write_text(f"""[alembic]
script_location = migrations
sqlalchemy.url = postgresql+psycopg2://vectordb:vectordb@localhost:5432/vectordb

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers = console

[logger_alembic]
level = INFO
handlers = console

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
""")

    print("Alembic configuration written.")


# Validate Project (placeholder)
def validate_project():
    print("Validating project structure...")
    required_files = [
        "app/main.py", "app/api/routes/__init__.py", "app/models/__init__.py",
        "app/services/__init__.py", "app/core/config.py"
    ]
    base_path = Path("c:/study/autogen/generated_project")
    for rel_path in required_files:
        if not (base_path / rel_path).exists():
            raise FileNotFoundError(f"Missing file: {rel_path}")
    print("All required files present.")


# Main Execution Pipeline
def execute_pipeline(requirements):
    try:
        print("=== Step 1: Analyze SRS Document ===")
        extracted = analyze_srs_document(requirements["srs_document"])
        print("SRS Document analyzed.")

        print("=== Step 2: Generate Code ===")
        generate_code(extracted)

        print("=== Step 3: Setup Environment ===")
        setup_environment()

        print("=== Step 4: Setup Database ===")
        setup_database()

        print("=== Step 5: Validate Project ===")
        validate_project()

        print("Pipeline executed successfully.")

    except Exception as e:
        print(f"Pipeline error: {e}")
        traceback.print_exc()


# Example usage:
# with open("path_to_srs.docx", "rb") as f:
#     srs_data = f.read()
# execute_pipeline({"srs_document": srs_data})
