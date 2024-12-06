from fastapi import APIRouter
from pydantic import BaseModel
from app.api.services.users import get_embeddings

class QueryData(BaseModel):
    path: str

router = APIRouter(prefix="/users", tags=["user"])


@router.post("/ask")
async def ask(q: QueryData):
    return {"output_data": get_embeddings(q.path)}