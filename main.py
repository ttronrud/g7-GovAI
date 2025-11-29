import asyncio
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import logging
from QueryCoordinator import QueryCoordinator
from QueryService import SearchService
from QueryState import QueryState
import time

app = FastAPI()

@app.get("/get-response/{query_id}")
def queryResult(
    query_id: int
):
    res = QueryCoordinator.get_instance().getQID(query_id)
    res['processing_stage'] = res['processing_stage'].name
    return JSONResponse(content={
        "query_id":query_id, 
        "res":res
    })

# submit a new proposal onto processing queue
@app.post("/submit/")
async def upload_text(
    file: UploadFile = File(...)
):
    try:
        content_bytes = await file.read()
        content_str = content_bytes.decode("utf-8")
    except:
        return JSONResponse(content={
            "query_id":-1
        }) 
    
    # create submission ID to link processing
    qid = QueryCoordinator.get_instance().addQID(content_str)
    return JSONResponse(content={
        "query_id":qid
    })

# Basic prototype main
# run localhost for demo-ing
# this back-end will be queried from a secured front-end
# and will not be exposed publicly
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, 
                        filename="logs/" + str(time.asctime().replace(":","-")) + "_log.txt", 
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("MAIN::Logging Initialized successfully")
    
    QueryCoordinator.get_instance()
    service = SearchService()
    QueryCoordinator.get_instance().addService(service)
    
    HOST = "localhost"
    PORT = 8002
    logging.info(f"MAIN::Setting IO to {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
    logging.info("MAIN::App service started by Uvicorn")
