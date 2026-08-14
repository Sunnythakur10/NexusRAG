from concurrent import futures
import sys
from pathlib import Path

import grpc

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

RPC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RPC_DIR.parent

# Add RPC directory so generated protobuf modules can be imported
sys.path.insert(0, str(RPC_DIR))

# Add project root so NexusRAG modules can be imported
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# gRPC generated modules
# ---------------------------------------------------------

import nexusrag_pb2
import nexusrag_pb2_grpc


# ---------------------------------------------------------
# NexusRAG imports
# ---------------------------------------------------------

from utils.lamport_clock import LamportClock
from memory.vector_store import query_similar_approved_lines


# ---------------------------------------------------------
# Retrieval Service
# ---------------------------------------------------------

class RetrievalService(nexusrag_pb2_grpc.RetrievalServiceServicer):

    def __init__(self):
        # Each distributed service maintains its own Lamport clock
        self.clock = LamportClock()

    def RetrieveSimilarLines(self, request, context):
        """
        Handle a retrieval request from the NexusRAG client.

        Lamport logic:
        1. Receive client timestamp.
        2. Update server clock.
        3. Perform ChromaDB retrieval.
        4. Increment server clock before sending response.
        5. Attach server timestamp to response.
        """

        # -------------------------------------------------
        # 1. Receive client event
        # -------------------------------------------------

        server_receive_timestamp = self.clock.receive(
            request.lamport_timestamp
        )

        print()
        print("======================================")
        print("[RPC Server] Request received")
        print(f"Client Lamport Timestamp : {request.lamport_timestamp}")
        print(f"Server Lamport Clock     : {server_receive_timestamp}")
        print("======================================")

        print(f"Character : {request.character_name}")
        print(f"Manga ID  : {request.manga_id}")
        print(f"Query     : {request.query_text}")
        print(f"Limit     : {request.limit}")

        try:

            # -------------------------------------------------
            # 2. Query ChromaDB
            # -------------------------------------------------

            rows = query_similar_approved_lines(
                character_name=request.character_name,
                manga_id=request.manga_id,
                query_text=request.query_text,
                limit=request.limit or 5,
            )

            print(
                f"[Retrieval Server] ChromaDB returned "
                f"{len(rows)} matching lines"
            )

            # -------------------------------------------------
            # 3. Local server event: prepare response
            # -------------------------------------------------

            server_response_timestamp = self.clock.tick()

            print(
                f"[RPC Server] Sending response | "
                f"Lamport Timestamp = {server_response_timestamp}"
            )

            # -------------------------------------------------
            # 4. Create gRPC response
            # -------------------------------------------------

            response = nexusrag_pb2.RetrievalResponse(
                lamport_timestamp=server_response_timestamp
            )

            for row in rows:
                response.lines.add(
                    panel_id=str(row.get("panel_id") or ""),
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
                f"[Retrieval Server] Returning "
                f"{len(response.lines)} matching lines"
            )

            return response

        except Exception as exc:

            print(
                f"[Retrieval Server] Error: {exc}"
            )

            context.set_code(
                grpc.StatusCode.INTERNAL
            )

            context.set_details(str(exc))

            # Even an error response gets the current logical time
            error_timestamp = self.clock.tick()

            return nexusrag_pb2.RetrievalResponse(
                lamport_timestamp=error_timestamp
            )


# ---------------------------------------------------------
# Start gRPC server
# ---------------------------------------------------------

def serve():

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=10
        )
    )

    nexusrag_pb2_grpc.add_RetrievalServiceServicer_to_server(
        RetrievalService(),
        server,
    )

    server.add_insecure_port(
        "[::]:50051"
    )

    server.start()

    print("======================================")
    print(" NexusRAG Retrieval gRPC Server")
    print(" Experiment 3: Lamport Clock Enabled")
    print(" Running on port 50051")
    print("======================================")

    server.wait_for_termination()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    serve()