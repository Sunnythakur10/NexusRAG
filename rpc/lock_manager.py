
import sys
import threading
import time
from concurrent import futures

import grpc

import nexusrag_pb2
import nexusrag_pb2_grpc


# ---------------------------------------------------------
# Experiment 5 resource definitions
# ---------------------------------------------------------

RESOURCES = {
    "rag_memory",
    "localization_output",
}


# ---------------------------------------------------------
# Lock Manager
# ---------------------------------------------------------

class LockManager(nexusrag_pb2_grpc.LockServiceServicer):

    def __init__(self, detect=False):

        self.detect = detect

        # Protects all shared lock-manager state.
        self.mutex = threading.Lock()

        # resource -> current holder
        self.locks = {
            "rag_memory": None,
            "localization_output": None,
        }

        # holder_id -> resource the holder is waiting for
        #
        # Example:
        # Node 1 -> localization_output
        # Node 2 -> rag_memory
        #
        # This represents:
        # Node 1 waits for the holder of localization_output
        # Node 2 waits for the holder of rag_memory
        self.wait_for = {}

        # One condition variable per resource.
        self.conditions = {
            resource: threading.Condition()
            for resource in RESOURCES
        }

    # -----------------------------------------------------
    # Validate resource
    # -----------------------------------------------------

    def _validate_resource(self, resource):

        if resource not in RESOURCES:
            raise ValueError(
                f"Unknown NexusRAG resource: {resource}"
            )

    # -----------------------------------------------------
    # Deadlock cycle detection
    # -----------------------------------------------------

    def _would_cycle(self, requester_id, resource):

        """
        Determine whether making requester_id wait for
        'resource' would create a circular wait.

        We follow:

            requester
                ↓
            requested resource
                ↓
            current owner
                ↓
            resource owner is waiting for
                ↓
            next owner
                ↓
              ...

        If the chain reaches requester_id, a cycle exists.
        """

        current_resource = resource
        visited_nodes = set()

        while True:

            owner = self.locks.get(current_resource)

            # Nobody owns this resource.
            if owner is None:
                return False

            # The requester would eventually wait for itself.
            if owner == requester_id:
                return True

            # Prevent infinite traversal.
            if owner in visited_nodes:
                return False

            visited_nodes.add(owner)

            # What resource is the current owner waiting for?
            current_resource = self.wait_for.get(owner)

            if current_resource is None:
                return False

    # -----------------------------------------------------
    # AcquireLock RPC
    # -----------------------------------------------------

    def AcquireLock(self, request, context):

        resource = request.resource_id
        holder = request.holder_id
        timestamp = request.timestamp

        try:
            self._validate_resource(resource)
        except ValueError as exc:

            return nexusrag_pb2.LockReply(
                granted=False,
                message=str(exc),
            )

        condition = self.conditions[resource]

        with condition:

            with self.mutex:

                owner = self.locks.get(resource)

                print(
                    f"[LockManager] Acquire request | "
                    f"Node-{holder} | "
                    f"Resource='{resource}' | "
                    f"Timestamp={timestamp} | "
                    f"Owner={owner}"
                )

                # -------------------------------------------------
                # Case 1: resource is free
                # -------------------------------------------------

                if owner is None:

                    self.locks[resource] = holder
                    self.wait_for.pop(holder, None)

                    print(
                        f"[LockManager] GRANTED | "
                        f"Node-{holder} -> '{resource}'"
                    )

                    return nexusrag_pb2.LockReply(
                        granted=True,
                        message="granted",
                    )

                # -------------------------------------------------
                # Case 2: requester already owns the resource
                # -------------------------------------------------

                if owner == holder:

                    print(
                        f"[LockManager] Node-{holder} already "
                        f"owns '{resource}'"
                    )

                    return nexusrag_pb2.LockReply(
                        granted=True,
                        message="already-held",
                    )

                # -------------------------------------------------
                # Case 3: deadlock detection mode
                # -------------------------------------------------

                if self.detect and self._would_cycle(
                    holder,
                    resource,
                ):

                    print(
                        f"[LockManager] DEADLOCK DETECTED: "
                        f"Node-{holder} -> '{resource}' "
                        f"(held by Node-{owner}) would close a cycle. "
                        f"Aborting Node-{holder}."
                    )

                    return nexusrag_pb2.LockReply(
                        granted=False,
                        message="deadlock-abort",
                    )

                # -------------------------------------------------
                # Case 4: resource is busy
                # -------------------------------------------------

                self.wait_for[holder] = resource

                print(
                    f"[LockManager] WAITING | "
                    f"Node-{holder} waiting for '{resource}' "
                    f"(held by Node-{owner})"
                )

            # -----------------------------------------------------
            # Simulation mode:
            # wait until the resource becomes available.
            #
            # Detection mode only reaches here when no cycle
            # would be formed, so waiting is also safe.
            # -----------------------------------------------------

            while True:

                with self.mutex:

                    owner = self.locks.get(resource)

                    if owner is None:

                        self.locks[resource] = holder
                        self.wait_for.pop(holder, None)

                        print(
                            f"[LockManager] GRANTED AFTER WAIT | "
                            f"Node-{holder} -> '{resource}'"
                        )

                        return nexusrag_pb2.LockReply(
                            granted=True,
                            message="granted-after-wait",
                        )

                # Wake periodically and re-check.
                condition.wait(timeout=1.0)

    # -----------------------------------------------------
    # ReleaseLock RPC
    # -----------------------------------------------------

    def ReleaseLock(self, request, context):

        resource = request.resource_id
        holder = request.holder_id

        try:
            self._validate_resource(resource)
        except ValueError as exc:

            return nexusrag_pb2.LockReply(
                granted=False,
                message=str(exc),
            )

        condition = self.conditions[resource]

        with condition:

            with self.mutex:

                owner = self.locks.get(resource)

                if owner != holder:

                    print(
                        f"[LockManager] RELEASE REJECTED | "
                        f"Node-{holder} does not own "
                        f"'{resource}'"
                    )

                    return nexusrag_pb2.LockReply(
                        granted=False,
                        message="not-owner",
                    )

                self.locks[resource] = None

                # If this node was waiting on another resource,
                # remove that wait entry.
                self.wait_for.pop(holder, None)

                print(
                    f"[LockManager] RELEASED | "
                    f"Node-{holder} -> '{resource}'"
                )

                # Wake workers waiting for this resource.
                condition.notify_all()

                return nexusrag_pb2.LockReply(
                    granted=True,
                    message="released",
                )


# ---------------------------------------------------------
# Server
# ---------------------------------------------------------

def serve():

    detect = (
        len(sys.argv) > 1
        and sys.argv[1].lower() == "detect"
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=20
        )
    )

    lock_manager = LockManager(
        detect=detect
    )

    nexusrag_pb2_grpc.add_LockServiceServicer_to_server(
        lock_manager,
        server,
    )

    server.add_insecure_port(
        "localhost:60100"
    )

    server.start()

    print("======================================")
    print(" NexusRAG Lock Manager")
    print(" Experiment 5: Deadlock Simulation")
    print(
        f" Deadlock Detection = {detect}"
    )
    print(" Running on localhost:60100")
    print("======================================")

    try:

        while True:
            time.sleep(86400)

    except KeyboardInterrupt:

        print(
            "\n[LockManager] Shutting down..."
        )

        server.stop(0)


if __name__ == "__main__":
    serve()
