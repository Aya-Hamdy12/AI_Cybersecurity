from fastapi import FastAPI, APIRouter
from dotenv import load_dotenv
import os
load_dotenv(".env")

base_router = APIRouter(
    prefix="/api/v1",
    tags=["base"],
)

@base_router.get("/")
async def welcome():
    app_name = os.getenv("APP_NAME")
    app_version = os.getenv("APP_VERSION")

    return {
        "app_name": app_name,
        "app_version": app_version,
        "message": f"Welcome to {app_name} API version {app_version}!"
        }
