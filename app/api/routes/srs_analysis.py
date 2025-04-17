from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ai_service import execute_pipeline
import logging
from typing import Dict

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = [
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/msword'
]

@router.post("/analyze")
def analyze_srs(file: UploadFile = File(...)):
    """
    Endpoint to analyze an SRS document and generate a FastAPI project.

    Args:
        file (UploadFile): The uploaded SRS document (.docx format)

    Returns:
        dict: Status and details of the project generation process

    Raises:
        HTTPException: If file format is invalid or processing fails
    """
    try:
        logger.info("Starting analysis of SRS document.")
        logger.info(f"Uploaded file name: {file.filename}")
        logger.info(f"Uploaded file content type: {file.content_type}")

        # Validate file type
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a Word document (.docx)"
            )

        if not file.filename.lower().endswith('.docx'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file extension. Please upload a .docx file"
            )

        # Read the uploaded file content
        content = file.file.read()

        # Prepare requirements dictionary
        requirements = {"srs_document": content}

        # Execute the pipeline to generate the project
        logger.info("Executing the project generation pipeline.")
        result = execute_pipeline(requirements)

        # Check for pipeline errors
        if result.get("errors"):
            error_details = result["errors"][0] if result["errors"] else "Unknown error occurred"
            logger.error(f"Pipeline failed: {error_details}")
            raise HTTPException(
                status_code=500,
                detail=f"Project generation failed: {error_details}"
            )

        # Check if all required steps were completed
        required_steps = {"srs_analysis", "node_setup", "project_structure"}
        completed_steps = set(result.get("steps_completed", []))
        if not required_steps.issubset(completed_steps):
            missing_steps = required_steps - completed_steps
            logger.error(f"Missing required steps: {missing_steps}")
            raise HTTPException(
                status_code=500,
                detail=f"Project generation incomplete. Missing steps: {missing_steps}"
            )

        logger.info("Project generation pipeline completed successfully.")
        return {
            "status": "success",
            "message": "Project generated successfully",
            "details": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during project generation", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during project generation: {str(e)}"
        )