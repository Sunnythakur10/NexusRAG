import sys
import threading
import time
from pathlib import Path

import grpc

RPC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RPC_DIR.parent

sys.path.insert(0, str(RPC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import nexusrag_pb2
import nexusrag_pb2_grpc

from utils.lamport_clock import LamportClock


# ---------------------------------------------------------
# NexusRAG backend servers
# ---------------------------------------------------------

BACKENDS = [
    ("NexusRAG-Backend-1", "localhost:60201"),
    ("NexusRAG-Backend-2", "localhost:60202"),
    ("NexusRAG-Backend-3", "localhost:60203"),
]


# Experiment requirement: 9 requests
NUM_REQUESTS = 9


# active[i] = number of currently in-flight requests
# assigned to backend i.
active = [0] * len(BACKENDS)

lock = threading.Lock()

# The load balancer also participates in the existing
# NexusRAG Lamport-enabled retrieval protocol.
clock = LamportClock()


# ---------------------------------------------------------
# Least Connections selection
# ---------------------------------------------------------

def pick_least_connections():
    """
    Select the backend with the fewest active connections.

    Important:
    The counter is incremented inside the same lock used to
    select the backend. This prevents two simultaneous
    requests from both seeing the same stale load.
    """

    with lock:

        idx = active.index(min(active))

        active[idx] += 1

        backend_name, backend_address = BACKENDS[idx]

        print(
            f"[LB] Routing -> {backend_name} "
            f"({backend_address}) | "
            f"active connections = {active}"
        )

        return idx


# ---------------------------------------------------------
# Release connection count
# ---------------------------------------------------------

def release_backend(idx):
    """
    Decrease the active connection count after the request
    completes, even if the RPC fails.
    """

    with lock:

        active[idx] -= 1

        backend_name, backend_address = BACKENDS[idx]

        print(
            f"[LB] Released -> {backend_name} "
            f"({backend_address}) | "
            f"active connections = {active}"
        )


# ---------------------------------------------------------
# Send one NexusRAG retrieval request
# ---------------------------------------------------------

def handle_request(req_id):

    idx = pick_least_connections()

    backend_name, backend_address = BACKENDS[idx]

    try:

        # Lamport local/send event.
        request_timestamp = clock.tick()

        print(
            f"[LB] Request {req_id} | "
            f"Lamport Timestamp = {request_timestamp} | "
            f"Backend = {backend_name}"
        )

        request = nexusrag_pb2.RetrievalRequest(
            character_name="Kira",
            manga_id="default",
            query_text=(
                "Even if the whole world comes crashing down "
                "around me, I won't back down an inch."
            ),
            limit=5,
            lamport_timestamp=request_timestamp,
        )

        with grpc.insecure_channel(
            backend_address
        ) as channel:

            stub = (
                nexusrag_pb2_grpc
                .RetrievalServiceStub(channel)
            )

            response = stub.RetrieveSimilarLines(
                request,
                timeout=15,
            )

        # Receive Lamport timestamp from backend.
        client_timestamp = clock.receive(
            response.lamport_timestamp
        )

        print(
            f"[LB] Request {req_id} completed | "
            f"Backend = {backend_name} | "
            f"Server Timestamp = "
            f"{response.lamport_timestamp} | "
            f"LB Clock = {client_timestamp} | "
            f"Results = {len(response.lines)}"
        )

        if response.lines:

            print(
                f"[LB] Request {req_id} retrieved: "
                f"{response.lines[0].final_output}"
            )

    except Exception as exc:

        print(
            f"[LB] Request {req_id} failed on "
            f"{backend_name}: {exc}"
        )

    finally:

        # ALWAYS release the connection count.
        release_backend(idx)


# ---------------------------------------------------------
# Main load-balancer experiment
# ---------------------------------------------------------

def main():

    print("==============================================")
    print(" NexusRAG Least Connections Load Balancer")
    print(" Experiment 6")
    print(" Strategy: Least Connections")
    print("==============================================")

    print(
        f"[LB] Backends: "
        f"{[address for _, address in BACKENDS]}"
    )

    print(
        f"[LB] Dispatching {NUM_REQUESTS} requests..."
    )

    threads = []

    for req_id in range(
        1,
        NUM_REQUESTS + 1,
    ):

        thread = threading.Thread(
            target=handle_request,
            args=(req_id,),
        )

        threads.append(thread)

        thread.start()

        # Same stagger as the mentor walkthrough.
        time.sleep(0.15)

    # Wait for all requests to finish.
    for thread in threads:
        thread.join()

    print()
    print("==============================================")
    print("[LB] All requests processed.")
    print(
        f"[LB] Final active connection counts: {active}"
    )
    print("==============================================")


if __name__ == "__main__":
    main()
