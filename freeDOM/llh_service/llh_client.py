"""
llh client:
packages messages, sends them to the llh service, and interprets replies
provides synchronous and asynchronous interfaces
"""

from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import uuid

import numpy as np
import zmq

import llh_cython


class LLHClient:

    slots = ["_max_hypos_per_batch", "_max_obs_per_batch", "_sock"]

    def __init__(self, req_addr, batch_size):
        self._init_socket(req_addr)
        self._max_hypos_per_batch = batch_size["n_hypos"]
        self._max_obs_per_batch = batch_size["n_observations"]

    @property
    def max_obs_per_batch(self):
        return self._max_obs_per_batch

    @property
    def max_hypos_per_batch(self):
        return self._max_hypos_per_batch

    # @profile
    def request_eval(self, x, mu, sig, req_id=""):
        """Request a single llh eval

        Parameters
        ----------
        x : shape (n_x,) numpy.ndarray of dtype float32
        mu : float32
        sig : float32
        req_id : optional
            Converted to str, and returned as such

        """

        if len(x) > self._max_obs_per_batch:
            raise ValueError(
                "len(x) must be <= the maximum batch size!"
                f" (In this case {self._max_obs_per_batch})"
            )

        # send a req_id string for development and debugging
        req_id_bytes = str(req_id).encode()

        theta_buff = np.empty(2, np.float32)
        theta_buff[0] = mu
        theta_buff[1] = sig

        # self._sock.send_multipart([req_id_bytes, x, theta_buff])
        llh_cython.dispatch_request(self._sock, req_id_bytes, x, theta_buff)

    def request_batch_eval(self, x, mus, sigs, req_id=""):
        """Request batch eval of llh(x|mu, sig) for all mus and sigs

        Parameters
        ----------
        x : shape (n_x,) numpy.ndarray of dtype float32
        mus : shape (n_hypos,) numpy.ndarray of dtype float32
        sigs : shape (n_hypos,) numpy.ndarray of dtype float32
        req_id : optional
            Converted to a string and returned as such

        """

        if len(x) * len(mus) > self._max_obs_per_batch:
            raise ValueError(
                "len(x)*n_hypos must be <= the maximum batch size!"
                f" (In this case {self._max_obs_per_batch})"
            )

        if len(mus) > self._max_hypos_per_batch:
            raise ValueError(
                "len(mus) must be <= the maximum hypothesis batch size!"
                f" (In this case {self._max_hypos_per_batch})"
            )

        req_id_bytes = str(req_id).encode()

        theta_buff = np.empty(2 * len(mus), np.float32)
        theta_buff[::2] = mus
        theta_buff[1::2] = sigs

        # self._sock.send_multipart([req_id_bytes, x, theta_buff])
        llh_cython.dispatch_request(self._sock, req_id_bytes, x, theta_buff)

    def recv(self, timeout=1000):
        if self._sock.poll(timeout, zmq.POLLIN) != 0:
            req_id, llh = self._sock.recv_multipart()
            return dict(req_id=req_id.decode(), llh=np.frombuffer(llh, np.float32))
        return None

    def eval_llh(self, x, mu, sig, timeout=1000):
        """ synchronous llh evaluation
            blocks until llh is ready
            raises RuntimeError on timeout

            Should not be used while asynchronous requests are in progress
        """

        req_id = uuid.uuid4().hex

        self.request_eval(x, mu, sig, req_id=req_id)

        reply = self.recv(timeout)
        if reply is None:
            raise RuntimeError("No reply from LLH service!")

        if reply["req_id"] != req_id:
            raise RuntimeError("uuid mismatch!")

        return reply["llh"][0]

    def _init_socket(self, req_addr):
        # pylint: disable=no-member
        ctxt = zmq.Context.instance()
        sock = ctxt.socket(zmq.DEALER)
        sock.connect(req_addr)

        self._sock = sock
        self._sock.setsockopt(zmq.LINGER, 0)
