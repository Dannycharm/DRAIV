from context_engine import ContextEngine
import zmq, time, json

def main():

    # From your CARLA printout
    K = [[640.0, 0.0, 640.0],
         [0.0, 640.0, 360.0],
         [0.0, 0.0,   1.0]]
    horizon_y = 360.0
    W, H = 1280, 720
    cam_height_m = 1.4 # 1.675549  # from sensor transform z (or use the exact camera height you attached)
    # If your camera was attached at ego local x=+1.2 m and Model 3 half-length ≈ 2.3 m:
    callibration_offset = 0.25
    cam_to_front_bumper_m = 1.1958897113800049 - callibration_offset # # measured from CARLA 
    
    # 1. Initialize ZeroMQ context and sockets
    zmq_context = zmq.Context() # Use a different variable name to avoid conflict with ContextEngine instance
    sub = zmq_context.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5555")  # Connect to perception publisher
    sub.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics (if any)
    # sub.setsockopt(zmq.RCVHWM, 10000)
    # sub.setsockopt(zmq.CONFLATE, 1)

    # To publish events from ContextEngine
    pub = zmq_context.socket(zmq.PUB)
    pub.bind("tcp://0.0.0.0:5575")  # Bind publisher for events (or connect to an events broker)

    # 2. Initialize ContextEngine
    context_engine = ContextEngine(log_to_file=True, log_file_path="drive_events.log", pub=pub)
    context_engine.configure_camera(K, horizon_y, (W, H), cam_height_m, cam_to_front_bumper_m)
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
