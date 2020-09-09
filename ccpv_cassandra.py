#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cassandra.cluster import Cluster, ExecutionProfile
from cassandra.policies import RetryPolicy

import time

# ===============================================================================


def retry_policy__on_read_timeout(query, consistency, required_responses,
                                  received_responses, data_retrieved,
                                  retry_num):
    if retry_num < 2:
        return (RetryPolicy.RETRY, None)
    return (RetryPolicy.RETHROW, None)


class cassandra:

    __cluster = None
    __session = None

    def __init__(self,
                 nodes=[
                     'f0c4.company', 'f0g5.company', 'r1c4.company',
                     'r1g5.company'
                 ],
                 keyspace='company_cg',
                 executor_threads=8,
                 request_timeout=10):
        self.__cluster = Cluster(nodes,
                                 connect_timeout=10,
                                 control_connection_timeout=10,
                                 executor_threads=executor_threads)
        self.__cluster.add_execution_profile(
            'default', ExecutionProfile(request_timeout=request_timeout))
        self.__cluster.default_retry_policy.on_read_timeout = retry_policy__on_read_timeout
        self.__session = self.__cluster.connect(keyspace)

    def session(self):
        return self.__session

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if self.__cluster:
            self.__cluster.shutdown()
