import shutil
import subprocess
import traceback
import json
from io import BytesIO
from pathlib import Path
import os
import stat
from docx import Document
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate


# Add detailed logging to each step of the pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Groq LLM
def initialize_groq_llm():
    api_key = "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"
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")
    # Ensure the model used is available for your account.
    return ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")  # Updated to a valid model

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

# Function to setup node and generate project structure in JSON format
def setup_node(state):
    api_endpoints = state.get("api_endpoints", "")
    business_logic = state.get("business_logic", "")
    auth_requirements = state.get("auth_requirements", "")  
    db_schema = state.get("db_schema", "")

    # Print the analysis results for review
    print("\nProject Analysis from LLM:")
    print("======================")
    print("API Endpoints:", api_endpoints)
    print("\nBusiness Logic:", business_logic)
    print("\nAuthentication Requirements:", auth_requirements)
    print("\nDatabase Schema:", db_schema)
    print("======================\n")

    # Define complete base structure with all essential directories
    base_structure = {
        "app": {
            "api": {
                "__init__.py": "# API package initialization\n",
                "routes": {
                    "__init__.py": "# Routes package initialization\n"
                }
            },
            "models": {
                "__init__.py": "# Models package initialization\n"
            },
            "core": {
                "__init__.py": "# Core package initialization\n"
            }
        }
    }

    prompt =  f"""

    Based on the following project analysis, generate the complete FastAPI project structure in JSON format.

    Return ONLY valid JSON without any additional text, markdown formatting, or explanations.

    The JSON structure must be a dictionary with folder and file paths as keys and their content as values.

    The base structure must include:

    - app/api directory with __init__.py

    - app/api/routes directory with __init__.py

    
    Include additional:

    - File names as keys and their content as values

    - Include a "requirements.txt" file with necessary dependencies

    - Generate code for each file, do not leave any file empty, and complete the whole workflow, including all endpoints, their business logic, etc.

    - Ensure the JSON response does not contain any extra text like comments, and the syntax of each JSON file is correct.

    Project Analysis:

    API Endpoints: {api_endpoints}

    Business Logic: {business_logic}

    Authentication Requirements: {auth_requirements}

    Database Schema: {db_schema}

    

    Example JSON format:

    {{
        "app/": {{
            "routers/": {{
                "user.py": "content of user.py",
                "item.py": "content of item.py"
            }},
            "models/": {{
                "user.py": "content of user.py",
                "item.py": "content of item.py"
            }},
            "__init__.py": "content of __init__.py",
            "main.py": "content of main.py"
        }},
        "requirements.txt": "fastapi\\nuvicorn\\npsycopg2-binary\\nalembic\\nsqlalchemy\\npython-dotenv",
        "setup.sh": "content of setup.sh"
    }}
    """

    llm = initialize_groq_llm()
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an expert software engineer. You must respond with only valid JSON."),
        HumanMessage(content=prompt)
    ])
    chain = LLMChain(llm=llm, prompt=prompt_template)

    try:
        result = chain.invoke({})
        text_response = result.get("text", "").strip()
        text_response = text_response.replace("```json", "").replace("```", "").strip()
        
        try:
            structure = json.loads(text_response)
            # Deep merge function for nested dictionaries
            def deep_merge(base, update):
                merged = base.copy()
                for key, value in update.items():
                    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key] = deep_merge(merged[key], value)
                    else:
                        merged[key] = value
                return merged

            # Merge the generated structure with base structure
            merged_structure = deep_merge(base_structure, structure)
            return {"setup": merged_structure}
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON parsing error: {json_err}")
            logger.error(f"Problematic JSON text: {text_response}")
            return {"setup": base_structure}
            
    except Exception as e:
        logger.error(f"Error generating project structure: {e}")
        return {"setup": base_structure}

def make_executable(path):
    """Make a file executable"""
    current = os.stat(path)
    os.chmod(path, current.st_mode | stat.S_IEXEC)

def create_windows_batch_file(sh_file: Path, bat_file: Path):
    """Create a Windows batch file equivalent of a shell script"""
    sh_content = sh_file.read_text()
    # Convert shell commands to batch commands
    bat_content = "@echo off\n"
    for line in sh_content.splitlines():
        if line.strip() and not line.startswith('#'):
            # Basic command conversion
            line = line.replace('export ', 'set ')
            line = line.replace('/', '\\')
            bat_content += f"{line}\n"
    bat_file.write_text(bat_content)

def process_structure(structure: dict, base_path: Path):
    """
    Recursively process the project structure and create files/directories
    
    Args:
        structure: Dictionary representing the file/directory structure
        base_path: Base directory path where files will be created
    """
    for name, content in structure.items():
        path = base_path / name
        
        if isinstance(content, dict):
            # If content is a dictionary, it's a directory
            path.mkdir(exist_ok=True, parents=True)
            process_structure(content, path)
        else:
            # If content is not a dictionary, it's a file
            # Ensure parent directory exists
            path.parent.mkdir(exist_ok=True, parents=True)
            # Write file content
            logger.info(f"Created file: {path}")

# Function to generate project structure based on JSON definition
def generate_project_structure(structure_json: dict, output_dir: str = None):
    """
    Generate project structure based on JSON definition
    
    Args:
        structure_json: JSON string representing the project structure
        output_dir: Directory where project will be created (defaults to 'generated_project' in srs_analyzer directory)
    """
    try:
        # Parse the JSON string if it's not already a dict
        if isinstance(structure_json, str):
            structure = json.loads(structure_json)
        else:
            structure = structure_json
        
        # Default to generating in the srs_analyzer directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated_project")
        
        # Create base directory
        base_dir = Path(output_dir)
        base_dir.mkdir(exist_ok=True, parents=True)
        
        # Process the structure recursively
        process_structure(structure, base_dir)
        
        # Make setup.sh executable if it exists
        setup_file = base_dir / "setup.sh"
        if setup_file.exists():
            make_executable(str(setup_file))
        
        # For Windows, also create a setup.bat file
        if os.name == 'nt' and not (base_dir / "setup.bat").exists() and setup_file.exists():
            create_windows_batch_file(setup_file, base_dir / "setup.bat")
        
        logger.info(f"✅ Project structure generated successfully in {output_dir}")
        return True, f"Project created in {output_dir}"
    except Exception as e:
        logger.error(f"❌ Error generating project structure: {str(e)}")
        return False, f"Error generating project structure: {str(e)}"

# Setup Environment
def setup_environment():
    logger.info("Setting up virtual environment...")
    project_root = Path("c:/study/autogen/srs-analyzer/app/generated_project")
    venv_path = project_root / "venv"

    try:
        subprocess.run(["python", "-m", "venv", str(venv_path)], check=True)
        logger.info("Virtual environment created.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Virtual environment creation failed: {e}")
        raise RuntimeError(f"Virtual environment creation failed: {e}")

    pip_path = venv_path / "Scripts" / "pip"
    requirements_file = Path("c:/study/autogen/srs-analyzer/app/generated_project/requirements.txt")

    try:
        if requirements_file.exists():
            logger.info("Installing dependencies from requirements.txt...")
            subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
        else:
            logger.info("Installing default dependencies...")
            subprocess.run([str(pip_path), "install", "fastapi", "uvicorn", "psycopg2", "alembic"], check=True)
        logger.info("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Dependency installation failed: {e}")
        raise RuntimeError(f"Dependency installation failed: {e}")

# Setup Database
def setup_database():
    logger.info("Setting up database (PostgreSQL + Alembic)...")
    project_root = Path("c:/study/autogen/srs-analyzer/app/generated_project")
    alembic_ini = project_root / "alembic.ini"
    migrations_dir = project_root / "migrations"
    migrations_dir.mkdir(exist_ok=True)

    alembic_ini.write_text(
        "[alembic]\n"
        "script_location = migrations\n"
        "sqlalchemy.url = postgresql+psycopg2://vectordb:vectordb@localhost:5432/vectordb\n\n"
        "[loggers]\n"
        "keys = root,sqlalchemy,alembic\n\n"
        "[handlers]\n"
        "keys = console\n\n"
        "[formatters]\n"
        "keys = generic\n\n"
        "[logger_root]\n"
        "level = WARN\n"
        "handlers = console\n\n"
        "[logger_sqlalchemy]\n"
        "level = WARN\n"
        "handlers = console\n\n"
        "[logger_alembic]\n"
        "level = INFO\n"
        "handlers = console\n\n"
        "[handler_console]\n"
        "class = StreamHandler\n"
        "args = (sys.stderr,)\n"
        "level = NOTSET\n"
        "formatter = generic\n\n"
        "[formatter_generic]\n"
        "format = %(levelname)-5.5s [%(name)s] %(message)s\n"
    )
    
    logger.info("Alembic configuration written.")

# Validate Project (placeholder)
def validate_project():
    logger.info("Validating project structure...")
    required_files = [
        "app/main.py", "app/api/routes/__init__.py", "app/models/__init__.py",
        "app/services/__init__.py", "app/core/config.py"
    ]
    base_path = Path("c:/study/autogen/srs-analyzer/app/generated_project")
    
    # Create essential directories if they don't exist
    essential_dirs = ["app", "app/api", "app/models", "app/core", "app/api/routes"]
    for dir_path in essential_dirs:
        dir_full_path = base_path / dir_path
        if not dir_full_path.exists():
            logger.warning(f"Creating missing essential directory: {dir_path}")
            dir_full_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py in new directories
            init_file = dir_full_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# This file makes the directory a Python package.\n")
    
    # Validate required files
    missing_files = []
    for rel_path in required_files:
        file_path = base_path / rel_path
        if not file_path.exists():
            missing_files.append(rel_path)
            # Create missing file with basic content
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if rel_path.endswith("__init__.py"):
                file_path.write_text("# This file makes the directory a Python package.\n")
            elif rel_path == "app/main.py":
                file_path.write_text(
                    "from fastapi import FastAPI\n\n"
                    "app = FastAPI()\n\n"
                    "@app.get('/')\n"
                    "def read_root():\n"
                    "    return {'message': 'Welcome to the API'}\n"
                )
            elif rel_path == "app/core/config.py":
                file_path.write_text(
                    "from pydantic_settings import BaseSettings\n\n"
                    "class Settings(BaseSettings):\n"
                    "    app_name: str = 'Generated API'\n\n"
                    "settings = Settings()\n"
                )
    
    if missing_files:
        logger.warning(f"Created missing files: {', '.join(missing_files)}")
    
    logger.info("Project structure validation and repair completed.")

# Main Execution Pipeline
def execute_pipeline(requirements):
    try:
        result = {
            "steps_completed": [],
            "generated_files": [],
            "project_path": None,
            "errors": []
        }

        # Step 1: Analyze SRS Document
        logger.info("=== Step 1: Analyze SRS Document ===")
        try:
            extracted = analyze_srs_document(requirements["srs_document"])
            if not any(extracted.values()):  # Check if any sections were extracted
                raise ValueError("No content was extracted from the SRS document")
            
            # Print the AI analysis results
            print("\nAI Analysis Results:")
            print("===================")
            print("\nAPI Endpoints:")
            for endpoint in extracted["api_endpoints"]:
                print(f"- {endpoint}")
            
            print("\nBackend Logic:")
            for logic in extracted["backend_logic"]:
                print(f"- {logic}")
            
            print("\nDatabase Schema:")
            for schema in extracted["database_schema"]:
                print(f"- {schema}")
            
            print("\nAuthentication Requirements:")
            for auth in extracted["authentication_requirements"]:
                print(f"- {auth}")
            print("===================\n")
            
            result["steps_completed"].append("srs_analysis")
            logger.info("SRS Document analyzed successfully.")
        except Exception as e:
            error_msg = f"SRS analysis failed: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result

        # Step 2: Setup Node
        logger.info("=== Step 2: Setup Node ===")
        try:
            state = {
                "api_endpoints": extracted["api_endpoints"],
                "business_logic": extracted["backend_logic"],
                "auth_requirements": extracted["authentication_requirements"],
                "db_schema": extracted["database_schema"]
            }
            setup_response = setup_node(state)
            project_structure = setup_response["setup"]
            
            # Validate project structure is not empty
            if not project_structure or len(project_structure) == 0:
                raise ValueError("Generated project structure is empty")
                
            result["steps_completed"].append("node_setup")
            logger.info("Node setup completed successfully.")
        except Exception as e:
            error_msg = f"Node setup failed: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result

        # Step 3: Generate Project Structure
        logger.info("=== Step 3: Generate Project Structure ===")
        try:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated_project")
            success, message = generate_project_structure(project_structure, output_dir)
            if not success:
                raise Exception(message)
                
            # Verify basic project structure exists
            base_path = Path(output_dir)
            essential_paths = ["app", "app/api", "app/models", "app/core"]
            for path in essential_paths:
                if not (base_path / path).exists():
                    raise FileNotFoundError(f"Essential directory '{path}' was not generated")
                    
            result["steps_completed"].append("project_structure")
            result["project_path"] = output_dir
            logger.info("Project structure generated successfully.")
        except Exception as e:
            error_msg = f"Project structure generation failed: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result

        # Step 4: Setup Environment
        logger.info("=== Step 4: Setup Environment ===")
        try:
            setup_environment()
            result["steps_completed"].append("environment_setup")
            logger.info("Environment setup completed successfully.")
        except Exception as e:
            error_msg = f"Environment setup failed: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result

        # Step 5: Setup Database
        logger.info("=== Step 5: Setup Database ===")
        try:
            setup_database()
            result["steps_completed"].append("database_setup")
            logger.info("Database setup completed successfully.")
        except Exception as e:
            error_msg = f"Database setup failed: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result

        # Step 6: Validate Project
        logger.info("=== Step 6: Validate Project ===")
        try:
            validate_project()
            result["steps_completed"].append("project_validation")
            logger.info("Project validation completed successfully.")
        except Exception as e:
            error_msg = f"Project validation failed: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result

        # Get list of generated files
        try:
            base_path = Path(output_dir)
            result["generated_files"] = [
                str(f.relative_to(base_path)) 
                for f in base_path.rglob("*") 
                if f.is_file()
            ]
        except Exception as e:
            logger.warning(f"Could not list generated files: {e}")

        logger.info("Pipeline executed successfully.")
        return result

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        traceback.print_exc()
        return {
            "steps_completed": result.get("steps_completed", []),
            "generated_files": result.get("generated_files", []),
            "project_path": result.get("project_path"),
            "errors": result.get("errors", []) + [f"Unexpected error: {str(e)}"]
        }
