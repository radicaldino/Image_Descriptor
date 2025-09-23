import base64
import json
import re
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.constants import OPENAI_API_KEY
from utils.models import SimilarProductRequest, ModelConfig


def create_llm_for_similar_products(cfg: Optional[ModelConfig] = None) -> ChatOpenAI:
    """
    Factory for a LangChain ChatOpenAI configured for similar product search.
    """
    cfg = cfg or ModelConfig()

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
    )
    return llm


def get_similar_products(request: SimilarProductRequest) -> Dict[str, Any]:
    """
    Get similar products from e-commerce platforms using the provided image.
    Returns both raw text and structured JSON data.
    """
    # Encode image to base64
    image_base64 = base64.b64encode(request.image_data).decode('utf-8')
    
    # Create the image URL format for OpenAI
    image_url = f"data:{request.content_type};base64,{image_base64}"
    
    # Create model config
    cfg = ModelConfig(
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens  # Fixed: use request.max_tokens instead of cfg.max_tokens
    )
    
    # Create LLM
    llm = create_llm_for_similar_products(cfg)
    
    # Create message with image
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": request.system_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ]
    )
    
    # Get response from LLM
    result = llm.invoke([message])
    raw_text = result.content
    
    # Extract JSON from the response
    json_data = extract_json_from_text(raw_text)
    
    return {
        "raw_text": raw_text,
        "json_data": json_data,
    }


def extract_json_from_text(text: str) -> Optional[Dict[Any, Any]]:
    """
    Extract JSON from text response, handling various formats.
    Returns normalized format with listings array.
    """
    # First, try to parse the entire text as JSON
    try:
        parsed = json.loads(text)
        return normalize_similar_products_response(parsed)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON blocks in the text using regex
    json_patterns = [
        r'```json\s*(\[.*?\])\s*```',  # JSON array in code blocks
        r'```json\s*(\{.*?\})\s*```',  # JSON object in code blocks
        r'```\s*(\[.*?\])\s*```',     # JSON array in generic code blocks
        r'```\s*(\{.*?\})\s*```',     # JSON object in generic code blocks
        r'(\[.*?\])',                  # Any array-like structure
        r'(\{[^{}]*"[^"]*"[^{}]*\})',  # Simple JSON object
        r'(\{.*?\})',                  # Any object-like structure
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                parsed = json.loads(match)
                return normalize_similar_products_response(parsed)
            except json.JSONDecodeError:
                continue
    
    return None


def normalize_similar_products_response(data: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize the LLM response to the expected format with listings array.
    """
    if data is None:
        return None
    
    # If it's already in the correct format with listings
    if isinstance(data, dict) and "listings" in data:
        return data
    
    # If it's an array of products, wrap it in listings
    if isinstance(data, list):
        return {"listings": data}
    
    # If it's a single product object, wrap it in listings array
    if isinstance(data, dict):
        return {"listings": [data]}
    
    return None
