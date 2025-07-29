from context_engine import ContextEngine
import zmq, time, json

def main():
     # 1. Initialize ZeroMQ context and sockets
    zmq_context = zmq.Context() # Use a different variable name to avoid conflict with ContextEngine instance
    sub = zmq_context.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5555")  # Connect to perception publisher
    sub.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics (if any)
    # sub.setsockopt(zmq.RCVHWM, 10000)
    sub.setsockopt(zmq.CONFLATE, 1)
    # To publish events from ContextEngine
    pub = zmq_context.socket(zmq.PUB)
    pub.bind("tcp://*:5560")  # Bind publisher for events (or connect to an events broker)

    # 2. Initialize your ContextEngine
    # context_engine = ContextEngine(log_to_file=True, log_file_path="drive_events.log", pub=pub)
    # Test without publisher
    # context_engine = ContextEngine()
    context_engine = ContextEngine(log_to_file=True, log_file_path="drive_events.log")
    print("ContextEngine ready, waiting for perception data...")

      # 3. Main loop for receiving and processing perception frames
    msg = {'t': 1217, 'ego_center': 640, 'ego_lane_offset_px': 210, 'drivable_ratio': 0.08509440104166667, 'objects': [{'id': 1289, 'cls': 'car', 'bbox': [502, 105, 639, 287], 'vx': 1.12213134765625, 'object_center_coord': [571.2091064453125, 196.7347412109375]}, {'id': 800, 'cls': 'car', 'bbox': [348, 164, 428, 225], 'vx': 3.0517578125e-05, 'object_center_coord': [388.4449462890625, 195.22134399414062]}, {'id': 1312, 'cls': 'car', 'bbox': [458, 152, 533, 219], 'vx': 0.7264404296875, 'object_center_coord': [495.8314208984375, 186.04840087890625]}, {'id': 1284, 'cls': 'traffic sign', 'bbox': [484, 117, 513, 144], 'vx': -1.080291748046875, 'object_center_coord': [499.0360412597656, 130.85580444335938]}, {'id': 1290, 'cls': 'traffic sign', 'bbox': [518, 112, 549, 141], 'vx': -1.2335205078125, 'object_center_coord': [534.320068359375, 126.63249206542969]}, {'id': 1282, 'cls': 'traffic sign', 'bbox': [455, 122, 476, 147], 'vx': -1.3895263671875, 'object_center_coord': [466.19573974609375, 134.87564086914062]}, {'id': 1325, 'cls': 'car', 'bbox': [415, 170, 430, 202], 'vx': -0.68450927734375, 'object_center_coord': [423.007568359375, 186.24142456054688]}]}
    try:
        print("Received frame:", msg)
        t0 = time.time()
        # Process the received message using the ContextEngine
        context_engine.process_frame(msg)
        # print("Received frame:", msg.get("t"))
        print(f"latency is: {(time.time() - t0) * 1000}ms")
    except KeyboardInterrupt:
        print("\nExiting. Closing ZeroMQ sockets and log file.")
    finally:
        # 4. Close log file and ZeroMQ sockets
        context_engine.close()
        sub.close()
        zmq_context.term()


if __name__ == "__main__":
    main()

