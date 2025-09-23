from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from utils.image_helper import *
from utils.models import *

# create a fastapi router with prefix /image
router = APIRouter(prefix="/image_describe")


@router.post("/describe")
async def describe_image(
    file: UploadFile = File(...),
    system_prompt: str = Form(...),
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 1000
):
    """
    Upload an image and get a description from the LLM based on the system prompt.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read the image data
        image_data = await file.read()
        
        # Create image description request
        image_request = ImageDescriptionRequest(
            image_data=image_data,
            filename=file.filename,
            content_type=file.content_type,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Get description from LLM
        description_result = get_image_description(image_request)
        
        return JSONResponse({
            "raw_text": description_result["raw_text"],
            "json_data": description_result["json_data"],
            "filename": file.filename,
            "model": model
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
