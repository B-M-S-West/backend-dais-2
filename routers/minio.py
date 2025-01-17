from fastapi import APIRouter, UploadFile, FastAPI
from minio import Minio
from dotenv import load_dotenv
from typing import List
import os
import asyncio
from aiohttp import ClientError

router = APIRouter(
    prefix="/minio",
    tags=["Minio"],
    responses={404: {"description": "Not found"}},
)

# Load environment variables from .env file
load_dotenv()

# Load environment variables
minio_host = os.environ.get('MINIO_ENDPOINT')
minio_access_key = os.environ.get('MINIO_ACCESS_KEY')
minio_secret_key = os.environ.get('MINIO_SECRET_KEY')
bucket = os.environ.get('MINIO_BUCKET')

# Establish connection to Minio
client = Minio(
    minio_host,
    access_key=minio_access_key,
    secret_key=minio_secret_key,
    secure=False
)

@router.post("/upload-file")
async def upload_file(file: UploadFile):
    try:
        client.put_object(
            bucket_name=bucket,
            object_name=file.filename,
            data = file.file,
            length = file.size
        )
    except ClientError as e:
        # Handle MinIO client errors
        print(f"Failed to upload file {file.filename}: {e}")

@router.post("/upload-files")
async def upload_files(files: List[UploadFile]):
    tasks = [upload_file(file) for file in files]
    await asyncio.gather(*tasks)

@router.post("/upload-folder")
async def upload_folder(folder_path: str):
    tasks = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'rb') as file:
                tasks.append(upload_file(UploadFile(file=file, filename=file_name)))
    await asyncio.gather(*tasks)