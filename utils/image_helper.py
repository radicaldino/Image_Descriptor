import base64
import json
import re
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.constants import OPENAI_API_KEY
from utils.models import ImageDescriptionRequest, ModelConfig


def create_llm_for_image(cfg: Optional[ModelConfig] = None) -> ChatOpenAI:
    """
    Factory for a LangChain ChatOpenAI configured for image processing.
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


def extract_json_from_text(text: str) -> Optional[Dict[Any, Any]]:
    """
    Extract JSON from text response, handling various formats.
    """
    try:
        # First, try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON blocks in the text using regex
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
        r'```\s*(\{.*?\})\s*```',     # JSON in generic code blocks
        r'(\{[^{}]*"[^"]*"[^{}]*\})',  # Simple JSON object
        r'(\{.*?\})',                  # Any object-like structure
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return None


def _normalize_code_and_desc(code: Optional[str], allowed: Dict[str, str]) -> Optional[Dict[str, str]]:
    if code is None:
        return None
    normalized_code = str(code).strip().upper()
    if normalized_code in allowed:
        description = allowed[normalized_code]
        return {
            "code": normalized_code,
            "description": description,
            "displayValue": f"{normalized_code} - {description}",
        }
    # Try description lookup (value to code)
    inv = {v.lower(): k for k, v in allowed.items()}
    lower = str(code).strip().lower()
    if lower in inv:
        mapped_code = inv[lower]
        description = allowed[mapped_code]
        return {
            "code": mapped_code,
            "description": description,
            "displayValue": f"{mapped_code} - {description}",
        }
    # Fallback: treat as freeform but still structure it
    return {
        "code": normalized_code,
        "description": normalized_code.title(),
        "displayValue": f"{normalized_code} - {normalized_code.title()}",
    }


def _first_non_empty(*values: Any) -> Optional[Any]:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


def normalize_json_to_schema(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Normalize arbitrary model output into the required schema:
    {
      "materialName": str,
      "unitOfMeasure": { code, description, displayValue },
      "materialType": { code, description, displayValue },
      "itemCategory": str,
      "itemSubCategory": str,
      "sku": str,
      "hsnCode": str
    }
    Returns None if input is None.
    """
    if data is None:
        return None

    # Allowed maps
    allowed_uom = {
        "KG": "Kilograms",
        "G": "Grams",
        "M": "Meters",
        "FT": "Feet",
        "LITERS": "Liters",
        "ML": "Milliliters",
        "PIECES": "Pieces",
        "PAIR": "Pair",
        "SET": "Set",
    }
    allowed_material_type = {
        "ROH": "Raw Material",
        "HALB": "Semi-Finished Product",
        "FERT": "Finished Product",
        "HAWA": "Trading Goods",
        "VERP": "Packaging Material",
        "NLAG": "Non-Stock Material",
    }

    # Common key variants
    def g(*keys: str) -> Optional[Any]:
        for k in keys:
            if k in data:
                return data[k]
        # also try case/underscore variants
        lowered = {str(k).replace(" ", "").replace("-", "").lower(): v for k, v in data.items()}
        for k in keys:
            norm = str(k).replace(" ", "").replace("-", "").lower()
            if norm in lowered:
                return lowered[norm]
        return None

    # materialName
    material_name = _first_non_empty(
        g("materialName", "material_name", "name", "productName", "title"),
    )
    if isinstance(material_name, dict):
        material_name = material_name.get("value") or material_name.get("name")
    if isinstance(material_name, (int, float)):
        material_name = str(material_name)

    # description (freeform)
    description = _first_non_empty(
        g(
            "description",
            "materialDescription",
            "productDescription",
            "details",
            "summary",
            "desc",
            "shortDescription",
            "longDescription",
            "productDetails",
            "product_details",
            "product details",
        ),
    )
    if isinstance(description, dict):
        description = description.get("value") or description.get("text") or description.get("description")
    if isinstance(description, (int, float)):
        description = str(description)

    # unitOfMeasure: can be object or string/code
    uom = g("unitOfMeasure", "unit_of_measure", "uom")
    if isinstance(uom, dict):
        uom_code = _first_non_empty(uom.get("code"), uom.get("value"), uom.get("name"))
    else:
        uom_code = uom
    unit_of_measure = _normalize_code_and_desc(uom_code, allowed_uom)

    # materialType
    mtype = g("materialType", "material_type", "type")
    if isinstance(mtype, dict):
        mtype_code = _first_non_empty(mtype.get("code"), mtype.get("value"), mtype.get("name"))
    else:
        mtype_code = mtype
    material_type = _normalize_code_and_desc(mtype_code, allowed_material_type)

    # itemCategory (string code expected)
    item_category = g("itemCategory", "item_category", "category")
    if isinstance(item_category, dict):
        item_category = _first_non_empty(item_category.get("code"), item_category.get("value"), item_category.get("name"))
    if isinstance(item_category, (int, float)):
        item_category = str(item_category)
    if isinstance(item_category, str):
        item_category = item_category.strip().upper()

    # itemSubCategory (freeform)
    item_sub_category = g("itemSubCategory", "item_sub_category", "subCategory", "subcategory")
    if isinstance(item_sub_category, dict):
        item_sub_category = _first_non_empty(item_sub_category.get("value"), item_sub_category.get("name"))
    if isinstance(item_sub_category, (int, float)):
        item_sub_category = str(item_sub_category)

    # sku, hsnCode
    sku = g("sku", "SKU", "productCode", "code")
    hsn = g("hsnCode", "hsn_code", "hsn")
    if isinstance(sku, (int, float)):
        sku = str(sku)
    if isinstance(hsn, (int, float)):
        hsn = str(hsn)

    normalized: Dict[str, Any] = {
        "materialName": material_name or "",
        "description": description or "",
        "unitOfMeasure": unit_of_measure or _normalize_code_and_desc(None, allowed_uom),
        "materialType": material_type or _normalize_code_and_desc(None, allowed_material_type),
        "itemCategory": item_category or "",
        "itemSubCategory": item_sub_category or "",
        "sku": sku or "",
        "hsnCode": hsn or "",
    }

    return normalized


def get_image_description(request: ImageDescriptionRequest) -> Dict[str, Any]:
    """
    Get image description from LLM using the provided image and system prompt.
    Returns both raw text and formatted JSON.
    """
    # Encode image to base64
    image_base64 = base64.b64encode(request.image_data).decode('utf-8')
    
    # Create the image URL format for OpenAI
    image_url = f"data:{request.content_type};base64,{image_base64}"
    
    # Create model config
    cfg = ModelConfig(
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    # Create LLM
    llm = create_llm_for_image(cfg)
    
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
    
    # Extract and normalize JSON from the response
    parsed = extract_json_from_text(raw_text)
    json_data = normalize_json_to_schema(parsed)

    # Fallback: try to derive description from raw text if missing
    if json_data is not None and (not json_data.get("description")):
        derived_desc = extract_description_from_text(raw_text)
        if derived_desc:
            json_data["description"] = derived_desc

    return {
        "raw_text": raw_text,
        "json_data": json_data,
    }


def extract_description_from_text(text: str) -> Optional[str]:
    """
    Heuristically extract a description from free-form text.
    Looks for:
      - A 'Description:' line
      - A 'Product Description:' line
      - A markdown heading '## Description' followed by a paragraph
    Returns the first plausible match.
    """
    if not text:
        return None

    # 1) Lines like: Description: ... (grab until end of line)
    line_patterns = [
        r"^\s*description\s*[:\-]\s*(.+)$",
        r"^\s*product\s*description\s*[:\-]\s*(.+)$",
        r"^\s*details\s*[:\-]\s*(.+)$",
        r"^\s*summary\s*[:\-]\s*(.+)$",
    ]
    for pat in line_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            candidate = m.group(1).strip()
            # Trim trailing markdown artifacts
            candidate = re.sub(r"\s*```.*$", "", candidate).strip()
            if candidate:
                return candidate

    # 2) Markdown section: ## Description\n<paragraph>
    section_patterns = [
        r"^\s*#+\s*description\s*\n+([\s\S]*?)(?:\n\s*#|\Z)",
        r"^\s*#+\s*product\s*description\s*\n+([\s\S]*?)(?:\n\s*#|\Z)",
    ]
    for pat in section_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            paragraph = m.group(1).strip()
            # Take first non-empty line/paragraph
            lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
            if lines:
                # Cap length to a reasonable size
                result = lines[0]
                if len(result) > 800:
                    result = result[:800].rstrip()
                return result

    # 3) As a last resort, take the first long-ish sentence from the text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for s in sentences:
        if len(s) >= 40:
            return s[:800].rstrip()

    return None
