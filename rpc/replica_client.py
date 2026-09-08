import sys
import time

import grpc

import nexusrag_pb2
import nexusrag_pb2_grpc


REPLICAS = {
    "A": 60301,
    "B": 60302,
    "C": 60303,
}


def save_value(replica_id, key, content):
    port = REPLICAS[replica_id]

    with grpc.insecure_channel(f"localhost:{port}") as channel:
        stub = nexusrag_pb2_grpc.ReplicaServiceStub(channel)

        response = stub.SaveValue(
            nexusrag_pb2.ValueUpdate(
                key=key,
                content=content,
                origin_replica=replica_id,
            ),
            timeout=3,
        )

        print(
            f"[CLIENT] WRITE -> Replica {replica_id} | "
            f"key={key} | content={content!r} | "
            f"accepted={response.accepted} | "
            f"ts={response.lamport_timestamp}"
        )


def get_value(replica_id, key):
    port = REPLICAS[replica_id]

    with grpc.insecure_channel(f"localhost:{port}") as channel:
        stub = nexusrag_pb2_grpc.ReplicaServiceStub(channel)

        response = stub.GetValue(
            nexusrag_pb2.ValueQuery(key=key),
            timeout=3,
        )

        return response


def print_state(replica_id, key):
    state = get_value(replica_id, key)

    print(
        f"[CLIENT] READ <- Replica {replica_id} | "
        f"key={state.key} | "
        f"content={state.content!r} | "
        f"ts={state.lamport_timestamp} | "
        f"origin={state.origin_replica}"
    )


if __name__ == "__main__":
    key = "localization::panel-42"

    print("=" * 70)
    print("NexusRAG Experiment 7")
    print("Eventual Consistency + Gossip Replication + LWW")
    print("=" * 70)

    print("\n[1] Two conflicting localization writes")
    print("-" * 70)

    # Writer 1
    save_value(
        "A",
        key,
        "I will never give up.",
    )

    # Writer 2
    save_value(
        "C",
        key,
        "I will never back down.",
    )

    print("\n[2] Immediate reads")
    print("-" * 70)

    print_state("A", key)
    print_state("B", key)
    print_state("C", key)

    print("\nWaiting for gossip convergence...")
    time.sleep(2)

    print("\n[3] Reads after convergence")
    print("-" * 70)

    print_state("A", key)
    print_state("B", key)
    print_state("C", key)

    print("\n" + "=" * 70)
    print("Experiment 7 complete")
    print("=" * 70)