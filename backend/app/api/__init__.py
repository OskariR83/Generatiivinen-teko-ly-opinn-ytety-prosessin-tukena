from fastapi import APIRouter
from app.api import routes, llm_routes

router = APIRouter()
router.include_router(routes.router)
router.include_router(llm_routes.router)
