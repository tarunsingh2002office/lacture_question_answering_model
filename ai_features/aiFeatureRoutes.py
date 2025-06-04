from fastapi import APIRouter
from ai_features.views.QuestionAnswerGenerationModel import QuestionAnswerGenerationModel

aiFeatureRoutes = APIRouter(prefix="/Ai_Features", tags=["AI"])


aiFeatureRoutes.add_api_route("/LactureQuestionAnswerGenerationModel", QuestionAnswerGenerationModel, methods=["POST"])
