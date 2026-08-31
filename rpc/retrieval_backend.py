import json
import random
import sys
import time
from concurrent import futures
from pathlib import Path

import grpc

RPC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RPC_DIR.parent

sys.path.insert(0, str(RPC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import nexusrag_pb2
import nexusrag_pb2_grpc

from utils.lamport_clock import LamportClock


SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "data"
    / "lb_retrieval_snapshot.json"
)


class NexusRAGBackend(
    nexusrag_pb2_grpc.RetrievalServiceServicer
):

    def __init__(self, name):

        self.name = name
        self.clock = LamportClock()

        if not SNAPSHOT_PATH.exists():
            raise FileNotFoundError(
                f"NexusRAG retrieval snapshot not found: "
                f"{SNAPSHOT_PATH}"
            )

        self.rows = json.loads(
            SNAPSHOT_PATH.read_text(
                encoding="utf-8"
            )
        )

        print(
            f"[{self.name}] Loaded "
            f"{len(self.rows)} retrieval records"
        )

    def RetrieveSimilarLines(
        self,
        request,
        context,
    ):
        # Maintain the Lamport behavior already used by
        # NexusRAG's Experiment 3 retrieval service.
        server_receive_timestamp = self.clock.receive(
            request.lamport_timestamp
        )

        # Variable backend workload is required by
        # Experiment 6 so that active connection counts
        # genuinely diverge over time.
        work_time = random.uniform(
            0.5,
            2.5,
        )

        print(
            f"[{self.name}] Handling request | "
            f"Character = {request.character_name} | "
            f"Lamport = {request.lamport_timestamp} | "
            f"Server Clock = {server_receive_timestamp} | "
            f"Work = {work_time:.1f}s"
        )

        time.sleep(work_time)

        response_timestamp = self.clock.tick()

        response = nexusrag_pb2.RetrievalResponse(
            lamport_timestamp=response_timestamp
        )

        # Return the same approved-line structure used by
        # the existing NexusRAG retrieval service.
        for row in self.rows[:request.limit or 5]:

            response.lines.add(
                panel_id=str(
                    row.get("panel_id") or ""
                ),
                original_japanese=str(
                    row.get("original_japanese") or ""
                ),
                final_output=str(
                    row.get("final_output") or ""
                ),
                created_at=int(
                    row.get("created_at") or 0
                ),
            )

        print(
            f"[{self.name}] Finished request | "
            f"Lamport = {response_timestamp} | "
            f"Results = {len(response.lines)}"
        )

        return response


def serve(port):

    name = f"NexusRAG-Backend-{port}"

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=20
        )
    )

    nexusrag_pb2_grpc.add_RetrievalServiceServicer_to_server(
        NexusRAGBackend(name),
        server,
    )

    server.add_insecure_port(
        f"localhost:{port}"
    )

    server.start()

    print("======================================")
    print(f" {name}")
    print(f" Listening on localhost:{port}")
    print(" Experiment 6: Least Connections")
    print("======================================")

    try:

        while True:
            time.sleep(86400)

    except KeyboardInterrupt:

        server.stop(0)


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: "
            "python rpc\\retrieval_backend.py <port>"
        )

        sys.exit(1)

    serve(int(sys.argv[1]))
