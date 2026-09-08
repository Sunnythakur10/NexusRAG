import sys
import time
import threading
from concurrent import futures

import grpc

sys.path.insert(0, ".")

from rpc import nexusrag_pb2
from rpc import nexusrag_pb2_grpc
from utils.lamport_clock import LamportClock


class NexusRAGReplica(nexusrag_pb2_grpc.ReplicaServiceServicer):
    """
    Experiment 7:
    NexusRAG Localization Memory Replication

    Each replica:
    - accepts local writes immediately
    - stores them locally
    - asynchronously gossips updates to peers
    - uses Lamport timestamps
    - resolves conflicts with Last-Write-Wins
    """

    def __init__(self, replica_id, port, peers):
        self.replica_id = replica_id
        self.port = port
        self.peers = peers

        self.clock = LamportClock()

        # key -> (content, lamport_timestamp, origin_replica)
        self.store = {}

        self.lock = threading.Lock()

        # Small delay makes the eventual-consistency window
        # reliably visible during the lab demonstration.
        self.gossip_delay = 0.5

    def _current_state(self, key):
        return self.store.get(key)

    def _apply_if_newer(self, key, content, timestamp, origin):
        """
        LWW comparison:
        (lamport_timestamp, origin_replica)
        """
        with self.lock:
            current = self.store.get(key)

            incoming_version = (timestamp, origin)

            if current is None:
                self.store[key] = (content, timestamp, origin)
                print(
                    f"[{self.replica_id}] APPLIED key={key} "
                    f"ts={timestamp} origin={origin} content={content!r}"
                )
                return True

            current_version = (current[1], current[2])

            if incoming_version > current_version:
                self.store[key] = (content, timestamp, origin)
                print(
                    f"[{self.replica_id}] APPLIED key={key} "
                    f"ts={timestamp} origin={origin} content={content!r}"
                )
                return True

            print(
                f"[{self.replica_id}] IGNORED stale key={key} "
                f"ts={timestamp} origin={origin}"
            )
            return False

    def _gossip(self, key, content, timestamp, origin):
        time.sleep(self.gossip_delay)

        for peer_id, peer_port in self.peers.items():
            try:
                with grpc.insecure_channel(f"localhost:{peer_port}") as channel:
                    stub = nexusrag_pb2_grpc.ReplicaServiceStub(channel)

                    update = nexusrag_pb2.ValueUpdate(
                        key=key,
                        content=content,
                        lamport_timestamp=timestamp,
                        origin_replica=origin,
                    )

                    stub.SyncUpdate(
                        update,
                        timeout=2,
                    )

                print(
                    f"[{self.replica_id}] GOSSIPED key={key} "
                    f"to={peer_id}"
                )

            except Exception as exc:
                print(
                    f"[{self.replica_id}] GOSSIP FAILED "
                    f"to={peer_id}: {exc}"
                )

    def SaveValue(self, request, context):
        """
        Accept local write immediately, then gossip asynchronously.
        """
        timestamp = self.clock.tick()

        self._apply_if_newer(
            request.key,
            request.content,
            timestamp,
            self.replica_id,
        )

        print(
            f"[{self.replica_id}] LOCAL WRITE accepted "
            f"key={request.key} ts={timestamp}"
        )

        thread = threading.Thread(
            target=self._gossip,
            args=(
                request.key,
                request.content,
                timestamp,
                self.replica_id,
            ),
            daemon=True,
        )
        thread.start()

        return nexusrag_pb2.SaveAck(
            accepted=True,
            replica=self.replica_id,
            lamport_timestamp=timestamp,
        )

    def SyncUpdate(self, request, context):
        """
        Receive an update from another replica.
        """
        self.clock.receive(request.lamport_timestamp)

        applied = self._apply_if_newer(
            request.key,
            request.content,
            request.lamport_timestamp,
            request.origin_replica,
        )

        return nexusrag_pb2.SaveAck(
            accepted=applied,
            replica=self.replica_id,
            lamport_timestamp=self.clock.now(),
        )

    def GetValue(self, request, context):
        state = self._current_state(request.key)

        if state is None:
            return nexusrag_pb2.ValueState(
                key=request.key,
                content="",
                lamport_timestamp=0,
                origin_replica="",
            )

        content, timestamp, origin = state

        return nexusrag_pb2.ValueState(
            key=request.key,
            content=content,
            lamport_timestamp=timestamp,
            origin_replica=origin,
        )


def serve(replica_id, port, peers):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    replica = NexusRAGReplica(
        replica_id=replica_id,
        port=port,
        peers=peers,
    )

    nexusrag_pb2_grpc.add_ReplicaServiceServicer_to_server(
        replica,
        server,
    )

    server.add_insecure_port(f"[::]:{port}")
    server.start()

    print("=" * 60)
    print(f"NexusRAG Replica {replica_id}")
    print(f"Port: {port}")
    print(f"Peers: {peers}")
    print("=" * 60)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python rpc\\replica_node.py "
            "<replica_id> <port>"
        )
        sys.exit(1)

    replica_id = sys.argv[1]
    port = int(sys.argv[2])

    ALL_REPLICAS = {
        "A": 60301,
        "B": 60302,
        "C": 60303,
    }

    peers = {
        rid: rport
        for rid, rport in ALL_REPLICAS.items()
        if rid != replica_id
    }

    serve(replica_id, port, peers)