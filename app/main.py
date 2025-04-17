import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.srs_analysis import router as srs_analysis_router

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add middleware to log requests and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Add CORS middleware to avoid potential issues with cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the SRS analysis router
app.include_router(srs_analysis_router, prefix="/api", tags=["SRS Analysis"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the SRS Analyzer API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)  # Changed port to 8080