from enum import Enum

# @Andrew - feel free to break PROCESSING into multiple stages
QueryState = Enum('QueryState',[('QUEUED',0),
                                ('PROCESSING_EMBEDDING',1),
                                ('PROCESSING_RERANKING',2),
                                ('PROCESSING_OUTPUT',3),
                                ('COMPLETE',4),
                                ('RETRIEVED',5)])