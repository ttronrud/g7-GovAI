from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import model
import uvicorn
import logging
from QueryCoordinator import QueryCoordinator
from QueryService import QueryService
import time

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/get-response/{query_id}")
def queryResult(
    query_id: int
):
    res = QueryCoordinator.get_instance().getQID(query_id)
    return JSONResponse(content={
        "query_id":query_id, 
        "res":res
    })

# submit a new proposal onto processing queue
@app.post("/submit/")
def upload_text(
    file: UploadFile = File(...)
):
    try:
        content_bytes = file.read()
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
    service = QueryService()
    QueryCoordinator.get_instance().addService(service)
    
    HOST = "localhost"
    PORT = 8002
    logging.info(f"MAIN::Setting IO to {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
    logging.info("MAIN::App service started by Uvicorn")