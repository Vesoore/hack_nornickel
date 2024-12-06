from fastapi import FastAPI
import uvicorn
from app.api.routers import index


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


app.include_router(index.router)

if __name__ == "__main__":
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=8001,
        reload=True
    )