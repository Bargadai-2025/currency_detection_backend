from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.apis.items import router
# from app.utils.config import MODEL_URL
# from ultralytics import YOLO

app =FastAPI(
    title='Currency Detection Backend',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router, prefix='/api')

@app.get('/health', tags=['Health'])
def health_check():
    return {'status': 'ok'}