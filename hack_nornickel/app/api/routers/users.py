from fastapi import APIRouter
from pydantic import BaseModel
from app.api.services.users import get_embeddings

class QueryData(BaseModel):
    path: str

router = APIRouter(prefix="/users", tags=["user"])

def tensor_to_list(tensor):
    return tensor.cpu().tolist()

@router.post("/ask")
async def ask(q: QueryData):
    embs = get_embeddings(q.path)
    response = []
    for data in embs:
        # Преобразуем тензор в список для JSON-сериализации
        embedding_list = tensor_to_list(data['embedding'])
        response.append({
            'page_num': data['page_num'],
            'embedding': embedding_list
        })
    return {"output_data": response}