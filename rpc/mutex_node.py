import argparse
import threading
import time
from concurrent import futures

import grpc

import sys
from pathlib import Path

RPC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RPC_DIR.parent

sys.path.insert(0, str(RPC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import nexusrag_pb2
import nexusrag_pb2_grpc

from utils.lamport_clock import LamportClock


RELEASED = "RELEASED"
WANTED = "WANTED"
HELD = "HELD"


class RicartAgrawalaNode(nexusrag_pb2_grpc.MutexServiceServicer):

    def __init__(self, node_id, peers):
        self.node_id = node_id
        self.peers = peers

        self.clock = LamportClock()

        self.state = RELEASED
        self.request_timestamp = None

        self.reply_count = 0
        self.deferred_requests = {}

        self.lock = threading.Lock()
        self.permission_event = threading.Event()

    def RequestCriticalSection(self, request, context):
        server_timestamp = self.clock.receive(
            request.lamport_timestamp
        )

        requester_id = request.node_id
        requester_timestamp = request.lamport_timestamp

        requester_priority = (
            requester_timestamp,
            requester_id,
        )

        print(
            f"[Node {self.node_id}] Request from Node "
            f"{requester_id} | "
            f"Request Timestamp = {requester_timestamp} | "
            f"Local Clock = {server_timestamp}"
        )

        with self.lock:

            my_priority = None

            if self.request_timestamp is not None:
                my_priority = (
                    self.request_timestamp,
                    self.node_id,
                )

            should_reply = (
                self.state == RELEASED
                or (
                    self.state == WANTED
                    and my_priority is not None
                    and requester_priority < my_priority
                )
            )

            if should_reply:
                response_timestamp = self.clock.tick()

                print(
                    f"[Node {self.node_id}] "
                    f"GRANT -> Node {requester_id}"
                )

                return nexusrag_pb2.MutexResponse(
                    granted=True,
                    lamport_timestamp=response_timestamp,
                )

            deferred_event = threading.Event()

            self.deferred_requests[requester_id] = (
                deferred_event
            )

            print(
                f"[Node {self.node_id}] "
                f"DEFER -> Node {requester_id}"
            )

        deferred_event.wait()

        with self.lock:
            self.deferred_requests.pop(
                requester_id,
                None,
            )

            response_timestamp = self.clock.tick()

        print(
            f"[Node {self.node_id}] "
            f"DEFERRED GRANT -> Node {requester_id}"
        )

        return nexusrag_pb2.MutexResponse(
            granted=True,
            lamport_timestamp=response_timestamp,
        )

    def request_critical_section(self):

        with self.lock:

            self.state = WANTED

            self.request_timestamp = self.clock.tick()

            self.reply_count = 0
            self.permission_event.clear()

            request_timestamp = self.request_timestamp

        print(
            f"[Node {self.node_id}] "
            f"WANTED | Timestamp = {request_timestamp}"
        )

        for peer_id, address in self.peers.items():

            if peer_id == self.node_id:
                continue

            thread = threading.Thread(
                target=self._request_peer,
                args=(
                    peer_id,
                    address,
                    request_timestamp,
                ),
                daemon=True,
            )

            thread.start()

        self.permission_event.wait()

        with self.lock:
            self.state = HELD

        print(
            f"[Node {self.node_id}] "
            f"HELD | All permissions received"
        )

    def _request_peer(
        self,
        peer_id,
        address,
        request_timestamp,
    ):

        try:

            with grpc.insecure_channel(address) as channel:

                stub = (
                    nexusrag_pb2_grpc
                    .MutexServiceStub(channel)
                )

                response = (
                    stub.RequestCriticalSection(
                        nexusrag_pb2.MutexRequest(
                            node_id=self.node_id,
                            lamport_timestamp=request_timestamp,
                        ),
                        wait_for_ready=True,
                    )
                )

                client_timestamp = self.clock.receive(
                    response.lamport_timestamp
                )

                if response.granted:

                    with self.lock:

                        self.reply_count += 1

                        print(
                            f"[Node {self.node_id}] "
                            f"GRANT received from Node {peer_id} | "
                            f"Clock = {client_timestamp} | "
                            f"Replies = "
                            f"{self.reply_count}/"
                            f"{len(self.peers) - 1}"
                        )

                        if (
                            self.reply_count
                            == len(self.peers) - 1
                        ):
                            self.permission_event.set()

        except Exception as exc:

            print(
                f"[Node {self.node_id}] "
                f"RPC error with Node {peer_id}: {exc}"
            )

    def release_critical_section(self):

        with self.lock:

            if self.state != HELD:
                raise RuntimeError(
                    "Cannot release critical section unless HELD"
                )

            self.state = RELEASED

            self.request_timestamp = None

            deferred_events = list(
                self.deferred_requests.values()
            )

            self.deferred_requests.clear()

            release_timestamp = self.clock.tick()

        print(
            f"[Node {self.node_id}] "
            f"RELEASED | "
            f"Deferred requests = "
            f"{len(deferred_events)} | "
            f"Clock = {release_timestamp}"
        )

        for event in deferred_events:
            event.set()

    def run_localization_critical_section(
        self,
        duration=5,
    ):

        self.request_critical_section()

        try:

            print(
                f"[Node {self.node_id}] "
                f">>> ENTERING NexusRAG "
                f"LOCALIZATION CRITICAL SECTION <<<"
            )

            print(
                f"[Node {self.node_id}] "
                f"Processing localization resource "
                f"for {duration} seconds..."
            )

            time.sleep(duration)

            print(
                f"[Node {self.node_id}] "
                f">>> LEAVING NexusRAG "
                f"LOCALIZATION CRITICAL SECTION <<<"
            )

        finally:

            self.release_critical_section()


def serve_node(
    node_id,
    port,
    peers,
):

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=20
        )
    )

    node = RicartAgrawalaNode(
        node_id,
        peers,
    )

    nexusrag_pb2_grpc.add_MutexServiceServicer_to_server(
        node,
        server,
    )

    server.add_insecure_port(
        f"[::]:{port}"
    )

    server.start()

    print("======================================")
    print(
        f" NexusRAG Ricart-Agrawala Node {node_id}"
    )
    print(
        f" Running on port {port}"
    )
    print(" Experiment 4: Mutual Exclusion")
    print("======================================")

    return server, node


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--node-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--port",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--request",
        action="store_true",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    peers = {
        1: "127.0.0.1:50061",
        2: "127.0.0.1:50062",
        3: "127.0.0.1:50063",
    }

    expected_ports = {
        1: 50061,
        2: 50062,
        3: 50063,
    }

    if args.port != expected_ports[args.node_id]:
        raise ValueError(
            f"Node {args.node_id} must use "
            f"port {expected_ports[args.node_id]}"
        )

    server, node = serve_node(
        args.node_id,
        args.port,
        peers,
    )

    try:

        if args.request:

            time.sleep(args.delay)

            node.run_localization_critical_section(
                args.duration
            )

        server.wait_for_termination()

    except KeyboardInterrupt:

        server.stop(0)


if __name__ == "__main__":
    main()
