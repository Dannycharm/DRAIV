from context_engine import ContextEngine
import zmq, time, json

def main():
     # 1. Initialize ZeroMQ context and sockets
    zmq_context = zmq.Context() # Use a different variable name to avoid conflict with ContextEngine instance
    sub = zmq_context.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5555")  # Connect to perception publisher
    sub.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics (if any)
    # sub.setsockopt(zmq.RCVHWM, 10000)
    # sub.setsockopt(zmq.CONFLATE, 1)
    
    # To publish events from ContextEngine
    pub = zmq_context.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:5575")  # Bind publisher for events (or connect to an events broker)

    # 2. Initialize your ContextEngine
    context_engine = ContextEngine(log_to_file=True, log_file_path="drive_events.log", pub=pub)
    print("ContextEngine ready, waiting for perception data...")

      # 3. Main loop for receiving and processing perception frames
    try:
        while True:
            # Receive one perception message
            msg = sub.recv_json()  # This correctly receives a single dictionary
            # print("Received frame:", msg)
            # t0 = time.time()
            # Process the received message using the ContextEngine
            context_engine.process_frame(msg)
            # print("Received frame:", msg.get("t"))
            # print(f"latency is: {(time.time() - t0) * 1000}ms")
    except KeyboardInterrupt:
        print("\nExiting. Closing ZeroMQ sockets and log file.")
    finally:
        # 4. Close log file and ZeroMQ sockets
        context_engine.close()
        sub.close()
        zmq_context.term()


if __name__ == "__main__":
    main()


