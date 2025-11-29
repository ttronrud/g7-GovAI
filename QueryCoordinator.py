import logging
from queue import Queue
import time
import json
import threading
import random
from QueryState import QueryState

static_instance = None

#Launch a daemon thread for a specific function
#return that thread handle
def launch(targ, args):
    t = threading.Thread(target=targ, args=args)
    t.daemon = True #make thread die with this program
    t.start()
    return t

#Thread function to process a service's process queue
def processQLoop(targ, monitor=None, timing = 0.05):
    while targ.isStillActive():
        time.sleep(timing)
        targ.processQueue(monitor)
        
class QueryCoordinator():
    def __init__(self, quiet_mode = False):
        #main Coordinator IO queue
        self.processQ = Queue()
        #memory dict to store QIDs -> data
        self.QIDdb = dict()
        #services to pass data to for processing
        self.services = []
        self.service_threads = []
        self.active = True  
        self.process_thread = None
        self.max_ongoing_queries = 10
        self.query_complete_timeout = 60
        self.process_times = []
        self.ptime_navg = 100
    def get_instance():
        global static_instance
        if static_instance == None:
            static_instance = QueryCoordinator()
        return static_instance
    
    def addQID(self, data):
        self.serviceRefresh()
        
        qid = random.randint(0, 1000000)
        while qid in self.QIDdb:
            qid = random.randint(0, 1000000)
        self.QIDdb[qid] = {'qid':qid, 
                            'time_init':time.time(), 'time_rdy':-1,
                            'data':data, 'output' : [], 'processing_stage':QueryState.QUEUED}
        logging.info(f"QCOORD::new QID generated: {qid}")
        for s in self.services:
            self.submitToService(s,self.QIDdb[qid])
        return qid
        
    def getQID(self, qid):
        if qid in self.QIDdb:
            if self.QIDdb[qid]['processing_stage'] == QueryState.COMPLETE:
                self.QIDdb[qid]['processing_stage'] = QueryState.RETRIEVED
            return self.QIDdb[qid]
        return None

    # update query state so that front-end can observe status
    def updateQIDState(self, qid, query_state):
        if qid in self.QIDdb:
            self.QIDdb[qid]['processing_stage'] = query_state
    
    def updateQIDData(self, qid, data):
        if qid in self.QIDdb:
            self.QIDdb[qid]['output'] += data
        
    
    #detects whether a thread has broken
    #by attempting to join with zero timeout
    #if thread has stopped, will join and return False
    def threadAlive(self, thr):
        if thr == None:
            return False
        thr.join(timeout=0.0)
        return thr.is_alive()
    
    #main process loop -- go through incoming data and
    #add it to it's QID's db entry, and mark as final if
    #all services have responded
    def processQueue(self, monitor):
        while not self.processQ.empty():
            qidDict = self.processQ.get()
            # do stuff
    
    #sends data to an async service, with a ref
    #to the main processQ to handle the response
    def submitToService(self, service, data):
        service.submit(data)
        logging.info(f"QCOORD::{data['qid']} submitted to service {service.getName()}")
    
    def endSystem(self):
        self.active = False
        
    def isStillActive(self):
        return self.active
        
    
    def serviceRefresh(self):
        if not self.threadAlive(self.process_thread):
            logging.info(f"QCOORD::Re-launching own process thread")
            self.process_thread = launch(processQLoop,(self, None,))
        
        self.service_threads = []
        for s,sth in zip(self.services, self.service_threads):
            if not self.threadAlive(sth):
                self.service_threads.append(launch(processQLoop,(service, self,)))
                logging.info(f"QCOORD::Service {service.getName()} process thread re-launched")
            else:
                self.service_threads.append(sth)
            
    
    #shuffle off the data to a service thread, and point the
    #response back to us
    def addService(self, service):
        if self.process_thread == None:
            logging.info(f"QCOORD::Launching Coordinator process thread")
            self.process_thread = launch(processQLoop,(self, None,))
        logging.info(f"QCOORD::Adding service {service.getName()}")
        self.services.append(service)
        self.service_threads.append(launch(processQLoop,(service, self,)))
        logging.info(f"QCOORD::Service {service.getName()} process thread launched")
