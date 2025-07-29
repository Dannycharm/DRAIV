

class ContextEngine:
    def __init__(self, log_to_file=False, log_file_path=None, pub=None):
        """
        Initialize the context engine.
        :param log_to_file: If True, enable logging of events to a file.
        :param log_file_path: Path to the log file (used if log_to_file is True).
        """
        self.log_to_file = log_to_file
        if log_to_file and log_file_path:
            # Open log file in write mode (existing content will be overwritten)
            self.log_file = open(log_file_path, 'w')
        else:
            self.log_file = None

        # State flags to remember if an event is currently active (to avoid repeat alerts)
        self.lane_departure_active = False
        self.pedestrian_ahead_active = False
        self.unsafe_distance_active = False

        # Log of past events (each entry is a dict with event type and start/end times)
        self.events_log = []

        # Threshold parameters (tunable as needed)
        self.LANE_OFFSET_THRESHOLD = 390         # px; beyond this lateral offset, consider lane departure
        self.DRIVABLE_RATIO_LANE_THRESH = 0.05    # if drivable area fraction falls below this, possibly off-lane
        self.VX_FRONT_THRESHOLD = 5.0            # px/frame; max lateral pixel shift to consider object in our lane (small = in path)
        self.DRIVABLE_RATIO_DISTANCE_THRESH = 0.05 # if drivable ratio below this with a car ahead, distance is unsafe
        self.FRONT_VEHICLE_DISTANCE_THRESH = 97.0
        # To publish events from ContextEngine
        self.pub = pub


    def process_frame(self, msg):
        """
        Process a single perception output frame (message).
        The msg should be a dict with keys: "t", "ego_lane_offset_px", "drivable_ratio", "objects".
        This method checks each condition and outputs/logs events as needed.
        """
        # Extract fields from the perception message
        t = msg.get("t")  # timestamp or frame index
        lane_offset = msg.get("ego_lane_offset_px", 0.0)
        if lane_offset == None:
            lane_offset = 0.0

        drivable_ratio = msg.get("drivable_ratio", 1.0)
        if drivable_ratio == None:
            drivable_ratio = 1.0
        objects = msg.get("objects", [])

        # --- Lane Departure Detection ---
        # Condition: car significantly off-center or minimal drivable area ahead (indicating off road)
        lane_departure_condition = (abs(lane_offset) > self.LANE_OFFSET_THRESHOLD)  or \
                                   (drivable_ratio < self.DRIVABLE_RATIO_LANE_THRESH)
        if lane_departure_condition:
            if not self.lane_departure_active:
                # Lane departure just started – trigger event
                self.lane_departure_active = True
                # Log the event start
                event_record = {"event": "lane_departure", "start": t, "end": None}
                self.events_log.append(event_record)
                self._output_warning(t, "Lane Departure")
        else:
            if self.lane_departure_active:
                # Lane departure condition ended – clear state and mark event end
                self.lane_departure_active = False
                # Update the last lane_departure event's end time
                for ev in reversed(self.events_log):
                    if ev["event"] == "lane_departure" and ev["end"] is None:
                        ev["end"] = t
                        break

        # --- Pedestrian Ahead Detection ---
        # Condition: any detected object classified as a pedestrian (in path)
        pedestrian_detected = any(obj.get("cls") == "pedestrian" for obj in objects)
        if pedestrian_detected:
            if not self.pedestrian_ahead_active:
                # New pedestrian hazard detected – trigger event
                self.pedestrian_ahead_active = True
                event_record = {"event": "pedestrian_ahead", "start": t, "end": None}
                self.events_log.append(event_record)
                self._output_warning(t, "Pedestrian Ahead")
        else:
            if self.pedestrian_ahead_active:
                # Pedestrian no longer present – clear state and mark event end
                self.pedestrian_ahead_active = False
                for ev in reversed(self.events_log):
                    if ev["event"] == "pedestrian_ahead" and ev["end"] is None:
                        ev["end"] = t
                        break

        # --- Unsafe Following Distance Detection ---
        # Determine if a vehicle is directly ahead (low lateral movement relative to us)
        vehicle_close_ahead = False
        for obj in objects:
            if obj.get("cls") in ["car", "truck", "bus", "motorcycle", "bicycle", "trailer", "other vehicle"]:  # vehicle types
                # vx is the horizontal pixel shift per frame (given by perception tracking)
                vx = obj.get("vx")
                if vx is None:
                    vx = 0.0
                bbox = obj.get("bbox")
                if bbox is not None and len(bbox) > 1:
                    front_vehicle_distance = bbox[1]
                else:
                    front_vehicle_distance = float("inf")  # default far value

                if (abs(vx) < self.VX_FRONT_THRESHOLD) and ( front_vehicle_distance < self.FRONT_VEHICLE_DISTANCE_THRESH) :
                    vehicle_close_ahead = True
                    break
        # Condition: a vehicle ahead AND drivable road area greatly reduced (front vehicle very close)
        unsafe_distance_condition = vehicle_close_ahead and (drivable_ratio < self.DRIVABLE_RATIO_DISTANCE_THRESH)
        if unsafe_distance_condition:
            if not self.unsafe_distance_active:
                # Start of unsafe following distance event
                self.unsafe_distance_active = True
                event_record = {"event": "unsafe_follow_distance", "start": t, "end": None}
                self.events_log.append(event_record)
                self._output_warning(t, "Unsafe Following Distance")
        else:
            if self.unsafe_distance_active:
                # Following distance back to safe – clear state and mark event end
                self.unsafe_distance_active = False
                for ev in reversed(self.events_log):
                    if ev["event"] == "unsafe_follow_distance" and ev["end"] is None:
                        ev["end"] = t
                        break

    def _output_warning(self, t, description):
        """
        Output a warning event (print to console and optionally log to file).
        """
        message = f"t={t}: {description} warning"
        # Real-time output: console print and publish to voice assistant
        print(message)
        if self.pub: # Check if pub socket exists
            try:
                self.pub.send_json({"type": "warning", "frame": t, "description": description, "message": message})
            except Exception as e:
                print(f"Error publishing warning: {e}") # Log error if publishing fails

        # Log to file if enabled
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()

    def close(self):
        """Close the log file and publisher socket if they were opened."""
        if self.log_file:
            self.log_file.close()
        if self.pub:
            pub.close()
