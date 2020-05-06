"""
 Dumbest possible broker

 Does nothing useful; it is for verifying that header frames 
 are passed through intermediate nodes correctly 
"""

__author__ = "Aaron Fienberg"

import sys

import zmq
import sys
import json


def main():
    with open("broker_params.json") as f:
        params = json.load(f)

    client_sock = zmq.Context.instance().socket(zmq.ROUTER)
    client_sock.bind(params["client_addr"])

    service_sock = zmq.Context.instance().socket(zmq.DEALER)
    service_sock.connect(params["service_addr"])

    poller = zmq.Poller()
    poller.register(client_sock, zmq.POLLIN)
    poller.register(service_sock, zmq.POLLIN)

    while True:
        events = poller.poll(100)
        for sock, event in events:
            if sock is service_sock:
                print("passing service reply")
                client_sock.send_multipart(service_sock.recv_multipart())
            elif sock is client_sock:
                print("passing client request")
                service_sock.send_multipart(client_sock.recv_multipart())


if __name__ == main():
    sys.exit(main())
