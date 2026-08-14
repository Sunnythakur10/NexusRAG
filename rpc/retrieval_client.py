import sys
from pathlib import Path

import grpc


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

RPC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RPC_DIR.parent

# Allow generated protobuf modules to be imported
sys.path.insert(0, str(RPC_DIR))

# Allow NexusRAG project modules to be imported
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Generated gRPC modules
# ---------------------------------------------------------

import nexusrag_pb2
import nexusrag_pb2_grpc


# ---------------------------------------------------------
# Lamport Clock
# ---------------------------------------------------------

from utils.lamport_clock import LamportClock


# ---------------------------------------------------------
# Retrieval Client
# ---------------------------------------------------------

class RetrievalClient:

    def __init__(self, host="localhost", port=50051):

        # gRPC connection
        self.channel = grpc.insecure_channel(
            f"{host}:{port}"
        )

        # gRPC service stub
        self.stub = nexusrag_pb2_grpc.RetrievalServiceStub(
            self.channel
        )

        # Each client maintains its own Lamport clock
        self.clock = LamportClock()

    def retrieve_similar_lines(
        self,
        character_name,
        manga_id,
        query_text,
        limit=5,
    ):
        """
        Send a retrieval request to the gRPC server.

        Lamport sequence:

        1. Client performs local event -> tick()
        2. Timestamp is attached to request
        3. Server processes request
        4. Server sends response timestamp
        5. Client receives response -> receive()
        """

        # -------------------------------------------------
        # 1. Client sends a request
        # -------------------------------------------------

        request_timestamp = self.clock.tick()

        print(
            f"[RPC Client] Sending request | "
            f"Lamport Clock = {request_timestamp}"
        )

        request = nexusrag_pb2.RetrievalRequest(
            character_name=character_name,
            manga_id=manga_id,
            query_text=query_text,
            limit=limit,
            lamport_timestamp=request_timestamp,
        )

        # -------------------------------------------------
        # 2. Perform RPC call
        # -------------------------------------------------

        response = self.stub.RetrieveSimilarLines(
            request
        )

        # -------------------------------------------------
        # 3. Receive server response
        # -------------------------------------------------

        client_timestamp = self.clock.receive(
            response.lamport_timestamp
        )

        print(
            f"[RPC Client] Response received | "
            f"Server Timestamp = {response.lamport_timestamp} | "
            f"Client Clock = {client_timestamp}"
        )

        # -------------------------------------------------
        # 4. Convert protobuf response to Python objects
        # -------------------------------------------------

        results = []

        for line in response.lines:

            results.append(
                {
                    "panel_id": line.panel_id,
                    "original_japanese": line.original_japanese,
                    "final_output": line.final_output,
                    "created_at": line.created_at,
                }
            )

        return results

    def close(self):
        """Close the gRPC channel."""

        self.channel.close()