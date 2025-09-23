from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from utils.similar_image_helper import *
from utils.models import *

# create a fastapi router with prefix /similar
router = APIRouter(prefix="/similar_products")


@router.post("/find")
async def find_similar_products(
    file: UploadFile = File(...),
    system_prompt: str = Form(...),
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 2000
):
    """
    Upload an image and find similar products available on e-commerce platforms.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read the image data
        image_data = await file.read()
        
        # Create similar products request
        similar_request = SimilarProductRequest(
            image_data=image_data,
            filename=file.filename,
            content_type=file.content_type,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Get similar products from LLM
        result = get_similar_products(similar_request)
        
        return JSONResponse({
            "raw_text": result["raw_text"],
            "json_data": result["json_data"],
            "filename": file.filename,
            "model": model
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
