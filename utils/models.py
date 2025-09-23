# pydantic models
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import UploadFile, File
from dataclasses import field








# ================================================
# =============== model configuration ===================
# ================================================

class ModelConfig(BaseModel):
    """
    Generic configuration for OpenAI chat models.
    For reasoning models (o4, o4-mini), set reasoning_effort to 'low' | 'medium' | 'high'.
    """
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    reasoning_effort: Optional[str] = None  # 'low' | 'medium' | 'high'
    extra_body: Dict[str, Any] = field(default_factory=dict)

# ================================================
# =============== image request models ===================
# ================================================


class ImageDescriptionRequest(BaseModel):
    image_data: bytes
    filename: str
    content_type: str
    system_prompt: str 
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1000











# ================================================
# =============== image response models ===================
# ================================================



class ImageDescriptionResponse(BaseModel):
    raw_text: str
    json_data: Optional[Dict[str, Any]] = None
    filename: str
    model: str


# ================================================
# =============== similar products models ===================
# ================================================

class SimilarProductRequest(BaseModel):
    image_data: bytes
    filename: str
    content_type: str
    system_prompt: str = "Identify all e-commerce platforms and websites where the product in this image is available. For each listing, provide the product's name, description, and its current price in India. Provide the data in a JSON array format, with each object containing the keys platform, name, description, link, and price."
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 2000


class ProductListing(BaseModel):
    source: str
    product: str
    price: str
    description: str
    link: str


class SimilarProductsResponse(BaseModel):
    raw_text: str
    json_data: Optional[Dict[str, Any]] = None
    filename: str
    model: str

