# import fastapi
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from utils import PORT
from apis.image_api import router as image_router
from apis.similar_image_api import router as similar_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.get("/")(lambda: {"message": "welcome to image description service current path is /image"})
# create a router
router = APIRouter(prefix="/image")
router.include_router(image_router)
app.include_router(router)

# create similar products router
similar_router_instance = APIRouter(prefix="/image")
similar_router_instance.include_router(similar_router)
app.include_router(similar_router_instance)


# add exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    print(f"HTTPException: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Bad Request"},
    )


# start the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)

