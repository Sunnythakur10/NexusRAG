import sys
import time

import grpc

import nexusrag_pb2
import nexusrag_pb2_grpc


LOCK_MANAGER = "localhost:60100"


RESOURCES = {
    1: ("rag_memory", "localization_output"),
    2: ("localization_output", "rag_memory"),
}


def acquire_lock(stub, resource, node_id):

    print(
        f"[Node {node_id}] REQUEST "
        f"'{resource}'"
    )

    response = stub.AcquireLock(
        nexusrag_pb2.LockRequest(
            resource_id=resource,
            holder_id=node_id,
            timestamp=int(time.time()),
        )
    )

    print(
        f"[Node {node_id}] "
        f"Lock response for '{resource}' | "
        f"Granted={response.granted} | "
        f"Message={response.message}"
    )

    return response.granted


def release_lock(stub, resource, node_id):

    print(
        f"[Node {node_id}] RELEASE "
        f"'{resource}'"
    )

    response = stub.ReleaseLock(
        nexusrag_pb2.LockRequest(
            resource_id=resource,
            holder_id=node_id,
            timestamp=int(time.time()),
        )
    )

    print(
        f"[Node {node_id}] "
        f"Release response | "
        f"Granted={response.granted} | "
        f"Message={response.message}"
    )


def run_worker(node_id):

    if node_id not in RESOURCES:
        raise ValueError(
            "node_id must be 1 or 2"
        )

    first_resource, second_resource = RESOURCES[
        node_id
    ]

    with grpc.insecure_channel(
        LOCK_MANAGER
    ) as channel:

        stub = nexusrag_pb2_grpc.LockServiceStub(
            channel
        )

        print("======================================")
        print(
            f" NexusRAG Deadlock Worker Node {node_id}"
        )
        print(" Experiment 5: Deadlock Simulation")
        print(
            f" Lock Manager: {LOCK_MANAGER}"
        )
        print("======================================")

        # -------------------------------------------------
        # First resource
        # -------------------------------------------------

        first_granted = acquire_lock(
            stub,
            first_resource,
            node_id,
        )

        if not first_granted:

            print(
                f"[Node {node_id}] "
                f"FIRST LOCK FAILED -> ABORT"
            )

            return

        print(
            f"[Node {node_id}] "
            f"HELD '{first_resource}'"
        )

        # Give the other worker time to acquire
        # its first resource.
        time.sleep(10)

        # -------------------------------------------------
        # Second resource
        # -------------------------------------------------

        print(
            f"[Node {node_id}] "
            f"Attempting SECOND LOCK "
            f"'{second_resource}'"
        )

        second_granted = acquire_lock(
            stub,
            second_resource,
            node_id,
        )

        if not second_granted:

            print(
                f"[Node {node_id}] "
                f"SECOND LOCK FAILED -> ABORT"
            )

            release_lock(
                stub,
                first_resource,
                node_id,
            )

            print(
                f"[Node {node_id}] "
                f"ABORT COMPLETE"
            )

            return

        # -------------------------------------------------
        # Both resources acquired
        # -------------------------------------------------

        print(
            f"[Node {node_id}] "
            f"ACQUIRED BOTH RESOURCES"
        )

        print(
            f"[Node {node_id}] "
            f">>> NEXUSRAG LOCALIZATION "
            f"CRITICAL OPERATION <<<"
        )

        time.sleep(10)

        print(
            f"[Node {node_id}] "
            f">>> LOCALIZATION OPERATION COMPLETE <<<"
        )

        # Release in reverse order.
        release_lock(
            stub,
            second_resource,
            node_id,
        )

        release_lock(
            stub,
            first_resource,
            node_id,
        )

        print(
            f"[Node {node_id}] "
            f"DONE"
        )


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python deadlock_worker.py <node_id>"
        )

        print(
            "Example: "
            "python deadlock_worker.py 1"
        )

        sys.exit(1)

    node_id = int(sys.argv[1])

    run_worker(node_id)
