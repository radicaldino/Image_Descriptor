# Image Description Service

A FastAPI-based service that provides AI-powered image description capabilities using OpenAI's vision models through LangChain integration.

## Features

- **Image Description**: Upload images and get detailed descriptions from OpenAI's vision models
- **Custom System Prompts**: Provide custom prompts to guide the image analysis
- **Model Configuration**: Support for various OpenAI vision models (gpt-4o, gpt-4o-mini)
- **Temperature & Token Control**: Fine-tune model behavior
- **Multiple Image Formats**: Support for common image formats (JPEG, PNG, WebP, etc.)

## API Endpoints

### Image Description Endpoints

#### `POST /image/image_describe/describe`
Upload an image and get a description from the LLM.

**Request:** Multipart form data with the following fields:
- `file`: Image file (required)
- `system_prompt`: Custom prompt for image analysis (optional, default: "Describe this image in detail.")
- `model`: OpenAI model to use (optional, default: "gpt-4o-mini")
- `temperature`: Model temperature (optional, default: 0.2)
- `max_tokens`: Maximum tokens in response (optional, default: 1000)

**Response:**
```json
{
  "description": "This image shows a beautiful sunset over a mountain range...",
  "filename": "sunset.jpg",
  "model": "gpt-4o-mini"
}
```

## Configuration

### Environment Variables

- `PORT`: Service port (default: 7321)
- `DEBUG`: Enable debug mode (default: false)
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Model Configuration

The service supports OpenAI vision models:
- `gpt-4o-mini` (default) - Faster, cost-effective
- `gpt-4o` - More detailed analysis

Both models support vision capabilities and can analyze images in various formats.

## Installation & Running

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
export PORT=7321
```

3. Run the service:
```bash
python main.py
```

### Docker

1. Build the image:
```bash
docker build -t maas .
```

2. Run the container:
```bash
docker run -p 7321:7321 -e OPENAI_API_KEY="your-key" maas
```

## Dependencies

- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `langchain-openai`: LangChain OpenAI integration
- `python-dotenv`: Environment variable management
- `openai`: OpenAI Python client
- `pydantic`: Data validation
- `python-multipart`: File upload support
- `pillow`: Image processing library

## Usage Examples

### Basic Image Description
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post("http://localhost:7321/image/image_describe/describe", 
        files={"file": f})
    description = response.json()["description"]
    print(description)
```

### Custom System Prompt
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post("http://localhost:7321/image/image_describe/describe", 
        files={"file": f},
        data={
            "system_prompt": "Analyze this image and identify any objects, people, and activities. Focus on the main subject and provide a detailed description.",
            "model": "gpt-4o",
            "temperature": 0.3,
            "max_tokens": 1500
        })
    description = response.json()["description"]
    print(description)
```

### Using curl
```bash
curl -X POST "http://localhost:7321/image/image_describe/describe" \
  -F "file=@image.jpg" \
  -F "system_prompt=Describe the colors and mood of this image" \
  -F "model=gpt-4o-mini"
```

## Architecture

The service is built with:
- **FastAPI**: Modern, fast web framework
- **LangChain**: Abstraction layer for LLM interactions
- **OpenAI**: Underlying AI model provider with vision capabilities
- **Pydantic**: Request/response validation
- **Pillow**: Image processing support

The service acts as a specialized interface to OpenAI's vision models, providing image analysis and description capabilities.
