import os # Import os for path manipulation if needed
import sys # For sys.stderr or sys.stdout if you want more controlled printing

class ContextEngine:
    def __init__(self, log_to_file=False, log_file_path=None, pub=None):
        """
        Initialize the context engine.
        :param log_to_file: If True, enable logging of events to a file.
        :param log_file_path: Path to the log file (used if log_to_file is True).
        :param pub: An optional publisher object (e.g., a ZeroMQ socket) to send warnings.
        """
        self.log_to_file = log_to_file
        self.log_file = None # Initialize to None
        if self.log_to_file and log_file_path:
            try:
                # Open log file in append mode ('a') so existing content is not overwritten
                # If 'w' (write mode) is strictly desired to overwrite each time, keep it.
                # For typical logging, 'a' (append) is more common.
                self.log_file = open(log_file_path, 'w')
            except IOError as e:
                print(f"Error opening log file {log_file_path}: {e}", file=sys.stderr)
                self.log_to_file = False # Disable logging to file if opening fails

        # State flags to remember if an event is currently active (to avoid repeat alerts)
        self.lane_departure_active = False
        self.pedestrian_ahead_active = False
        self.unsafe_distance_active = False

        # Log of past events (each entry is a dict with event type and start/end times)
        self.events_log = []

        # Threshold parameters (tunable as needed)
        self.LANE_OFFSET_THRESHOLD = 390       # px; beyond this lateral offset, consider lane departure
        self.DRIVABLE_RATIO_LANE_THRESH = 0.07  # if drivable area fraction falls below this, possibly off-lane
        self.VX_FRONT_THRESHOLD = 5.0          # px/frame; max lateral pixel shift to consider object in our lane (small = in path)
        self.FRONT_VEHICLE_DISTANCE_THRESH = 535.0 # remember that image ccorrdinates increase downward
        self.LANE_WIDTH_THRESH=800

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
        lane_offset = msg.get("ego_lane_offset_px") # Get without default for explicit None check below
        drivable_ratio = msg.get("drivable_ratio")
        if drivable_ratio is None: # Redundant if msg.get has a default, but harmless
            drivable_ratio = 0.0
        objects = msg.get("objects", [])

        lane_width = msg.get("lane_width")
        if lane_width is None:
            lane_width = 0

        # # --- Lane Departure Detection ---
        # # A lane width greater than the variable threshold indicates that a lane line was detected on both sides
        # if lane_offset is not None and lane_width > self.LANE_WIDTH_THRESH:
        #     # Condition: car significantly off-center or minimal drivable area ahead (indicating off road)
        #     lane_departure_condition = (abs(lane_offset) > self.LANE_OFFSET_THRESHOLD) or \
        #                 (drivable_ratio < self.DRIVABLE_RATIO_LANE_THRESH)
        #     if lane_departure_condition:
        #         if not self.lane_departure_active:
        #             # Lane departure just started – trigger event
        #             self.lane_departure_active = True
        #             # Log the event start
        #             event_record = {"event": "lane_departure", "start": t, "end": None}
        #             self.events_log.append(event_record)
        #             self._output_warning(t, "Lane Departure")
        #     else:
        #         if self.lane_departure_active:
        #             # Lane departure condition ended – clear state and mark event end
        #             self.lane_departure_active = False
        #             # Update the last lane_departure event's end time
        #             for ev in reversed(self.events_log):
        #                 if ev["event"] == "lane_departure" and ev["end"] is None:
        #                     ev["end"] = t
        #                     break

        # --- Pedestrian Ahead Detection ---
        # Condition: any detected object classified as a pedestrian (in path)
        pedestrian_detected = any(obj.get("cls") == "pedestrian_on_road" for obj in objects)
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
        ROI_CENTER = 640          # adjust if img width != 1280
        ROI_HALF   = 120          # +/- 120 px band around center
        Y_THRESH   = self.FRONT_VEHICLE_DISTANCE_THRESH  # 535 by default

        closest_y   = -1
        closest_id  = None
        vehicle_types = {"car","truck","bus","motorcycle","bicycle","rider","trailer","other vehicle"}

        for obj in objects:
            if obj.get("cls") not in vehicle_types:
                continue
            bbox = obj.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
                continue

            x1, y1, x2, y2 = bbox
            cx = 0.5 * (x1 + x2)

            # only consider vehicles roughly in our lane (center band)
            if (ROI_CENTER - ROI_HALF) <= cx <= (ROI_CENTER + ROI_HALF):
                # keep the CLOSEST one by bottom y (bigger y => closer in perspective)
                if y2 > closest_y:
                    closest_y  = y2
                    closest_id = obj.get("id")

        # decide unsafe based on the closest-in-ROI vehicle
        vehicle_close_ahead = (closest_y >= Y_THRESH) if closest_y >= 0 else False

        # optional debug: print the proxy distance for the best candidate
        # if closest_y >= 0:
            # print(f"Distance proxy (y2) = {int(closest_y)}  frame={t}  id={closest_id}")

        # state machine: alert on transition only; also print resolution
        if vehicle_close_ahead:
            if not self.unsafe_distance_active:
                self.unsafe_distance_active = True
                self.events_log.append({"event": "unsafe_follow_distance", "start": t, "end": None})
                self._output_warning(t, "Unsafe Following Distance")
        else:
            if self.unsafe_distance_active:
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
        #print(message)
        if self.pub: # Check if pub socket exists
            try:
                # Assuming self.pub has a send_json method
                self.pub.send_json({"type": "warning", "frame": t, "description": description, "message": message})
            except Exception as e:
                print(f"Error publishing warning: {e}", file=sys.stderr) # Log error if publishing fails

        # Log to file if enabled
        if self.log_file:
            try:
                self.log_file.write(message + "\n")
                self.log_file.flush() # Ensure data is written to disk immediately
            except IOError as e:
                print(f"Error writing to log file: {e}", file=sys.stderr)
                self.log_to_file = False # Disable logging if an error occurs

    def close(self):
        """Close the log file and publisher socket if they were opened."""
        if self.log_file:
            try:
                self.log_file.close()
            except Exception as e:
                print(f"Error closing log file: {e}", file=sys.stderr)
        if self.pub:
            try:
                self.pub.close()
            except Exception as e:
                print(f"Error closing publisher: {e}", file=sys.stderr)
