from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ai_service import execute_pipeline
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/analyze-srs")
def analyze_srs(file: UploadFile = File(...)):
    """
    Endpoint to analyze an SRS document and generate a FastAPI project.

    Args:
        file (UploadFile): The uploaded SRS document.

    Returns:
        dict: Status of the project generation process.
    """
    try:
        logger.info("Starting analysis of SRS document.")
        logger.info(f"Uploaded file name: {file.filename}")
        logger.info(f"Uploaded file content type: {file.content_type}")

        # Read the uploaded file content
        content = file.file.read()

        # Prepare requirements dictionary
        requirements = {"srs_document": content}

        # Execute the pipeline to generate the project
        logger.info("Executing the project generation pipeline.")
        execute_pipeline(requirements)

        logger.info("Project generation pipeline completed successfully.")
        return {"status": "success", "message": "Project generated successfully."}
    except Exception as e:
        logger.error(f"Error during project generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during project generation: {str(e)}")