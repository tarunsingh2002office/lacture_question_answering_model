from fastapi import APIRouter
from ai_features.aiFeatureRoutes import aiFeatureRoutes

api_router = APIRouter()

api_router.include_router(aiFeatureRoutes)