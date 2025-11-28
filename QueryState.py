from enum import Enum

# @Andrew - feel free to break PROCESSING into multiple stages
QueryState = Enum('QueryState',['QUEUED',
                                'PROCESSING_EMBEDDING',
                                'PROCESSING_RERANKING',
                                'PROCESSING_OUTPUT',
                                'COMPLETE',
                                'RETRIEVED'])