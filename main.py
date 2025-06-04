from fastapi import FastAPI
from core.routes import api_router

app = FastAPI()

app.include_router(api_router)



#uvicorn main:app --host 0.0.0.0 --port 8000 --reload