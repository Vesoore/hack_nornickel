from fastapi import APIRouter
from app.api.routers import users

router = APIRouter(prefix="/api")
router.include_router(users.router)

