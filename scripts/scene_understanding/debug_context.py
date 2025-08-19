import os # Import os for path manipulation if needed
import sys # For sys.stderr or sys.stdout if you want more controlled printing
import time
import math

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
                self.log_file.write("frame,description,sent_time\n")
            except IOError as e:
                print(f"Error opening log file {log_file_path}: {e}", file=sys.stderr)
                self.log_to_file = False # Disable logging to file if opening fails

        # State flags to remember if an event is currently active (to avoid repeat alerts)
        self.lane_departure_active = False
        self.pedestrian_ahead_active = False
        self.unsafe_distance_active = False

        # camera settings
        self.horizon_y = 360.0 # horizon line y-coordinate in pixels (adjust as needed)
        self.img_h = 720.0 # image height in pixels (adjust as needed)
        self.fov_deg = 90.0 # camera field of view in degrees (adjust as needed)
        self.height_m = 1.4 # camera height in meters above the ground (adjust as needed)

        # Since I don't know the exact lead rear overhang, I use a guess by class
        self.rear_overhang_by_cls = {
            "car": 0.9, "truck": 1.2, "bus": 1.5, "trailer": 1.2,
            "motorcycle": 0.2, "bicycle": 0.1, "rider": 0.2, "other vehicle": 1.0
        }


        # ---- Camera model (filled by configure_camera) ----
        self.fx = None; self.fy = None; self.cx = None; self.cy = None
        self.horizon_y = None
        self.img_h = None
        self.cam_height_m = None          # camera height above ground
        self.cam_to_front_bumper_m = 0.0  # along ego forward axis



        # Log of past events (each entry is a dict with event type and start/end times)
        self.events_log = []

        # Threshold parameters (tunable as needed)
        # self.LANE_OFFSET_THRESHOLD = 390       # px; beyond this lateral offset, consider lane departure
        # self.DRIVABLE_RATIO_LANE_THRESH = 0.07  # if drivable area fraction falls below this, possibly off-lane
        self.VX_FRONT_THRESHOLD = 5.0          # px/frame; max lateral pixel shift to consider object in our lane (small = in path)
        # self.FRONT_VEHICLE_DISTANCE_THRESH = 535.0 # remember that image ccorrdinates increase downward
        self.LANE_WIDTH_THRESH=800
        self.UNSAFE_THW_S  = 2.0
        self.SPEED_EPS     = 0.1

        # To publish events from ContextEngine
        self.pub = pub

        # other state variables
        self.fps = 30 # frames per second (adjust as needed)
        self.vehicle_states = {} # Stores {track_id: (timestamp_s, distance_m)}


    # --- call this once with values you extracted from CARLA ---
    def configure_camera(self, K, horizon_y, img_size, cam_height_m, cam_to_front_bumper_m):
        """
        K: 3x3 intrinsics (numpy-like) [[fx,0,cx],[0,fy,cy],[0,0,1]]
        horizon_y: pixel row of horizon (float)
        img_size: (W, H)
        cam_height_m: camera height above ground in meters
        cam_to_front_bumper_m: distance from camera center to ego front bumper along forward axis
        """
        self.fx = float(K[0][0]); self.fy = float(K[1][1])
        self.cx = float(K[0][2]); self.cy = float(K[1][2])
        self.horizon_y = float(horizon_y)
        self.img_h = float(img_size[1])
        self.cam_height_m = float(cam_height_m)
        self.cam_to_front_bumper_m = float(cam_to_front_bumper_m)

    # pixel row -> ground distance (camera center to ground point)
    def _ground_Z_from_y(self, y_pix: float) -> float:
        # Guard rails
        if self.fy is None or self.horizon_y is None or self.cam_height_m is None:
            return float('nan')
        denom = max(y_pix - self.horizon_y, 1.0)  # pixels below horizon
        Z = (self.fy * self.cam_height_m) / denom
        return max(Z, 0.0)

    # (optional) convenience to get pixel row for a given ground range Z (for gating)
    def _y_from_Z(self, Z: float) -> float:
        return self.horizon_y + (self.fy * self.cam_height_m) / max(Z, 0.01)

        #----------------------------------------------------
    def _rough_distance_from_y(self, y2, img_h):
        fy = (0.5 * img_h) / math.tan(math.radians(self.fov_deg * 0.5))
        denom = max(y2 - self.horizon_y, 1.0) # Denominator is distance below horizon
        Z = (fy * self.height_m) / denom
        return max(Z, 0.1)

    def process_frame(self, msg):
        """
        Process a single perception output frame (message).
        The msg should be a dict with keys: "t", "ego_lane_offset_px", "drivable_ratio", "objects".
        This method checks each condition and outputs/logs events as needed.
        """
        # Extract fields from the perception message
        t = msg.get("t")  # timestamp or frame index
        objects = msg.get("objects", [])

        # Uncomment these lines if you want to use lane offset and drivable ratio butit's not robust enough
        # lane_offset = msg.get("ego_lane_offset_px") # Get without default for explicit None check below
        # drivable_ratio = msg.get("drivable_ratio")
        # if drivable_ratio is None: # Redundant if msg.get has a default, but harmless
        #     drivable_ratio = 0.0

        # lane_width = msg.get("lane_width")
        # if lane_width is None:
            # lane_width = 0

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
                self._output_warning(t, "pedestrian_ahead")
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
        t = msg.get("t")  # this is a frame index; we turn it into seconds via fps
        objects = msg.get("objects", [])
        ego_speed = float(msg.get("ego_speed_mps") or 0.0)

        ROI_CENTER = self.cx if self.cx is not None else 640.0
        ROI_HALF   = 75  # keep your tuned value (or make it depend on curvature later)

        # Instead of a hard-coded Y_THRESH pixel, derive from a meter gate if you want:
        # e.g., treat anything within 25 m as "close candidate"
        gate_m = 25.0
        Y_THRESH = self._y_from_Z(gate_m) if self.fy else 535.0

        closest = None  # (y2, id, cls)
        vehicle_types = {"car","truck","bus","motorcycle","bicycle","rider","trailer","other vehicle"}

        for obj in objects:
            if obj.get("cls") not in vehicle_types:
                continue
            bbox = obj.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
                continue
            x1, y1, x2, y2 = bbox
            cx = 0.5 * (x1 + x2)

            # in-path gate
            if (ROI_CENTER - ROI_HALF) <= cx <= (ROI_CENTER + ROI_HALF):
                if closest is None or y2 > closest[0]:
                    closest = (y2, obj.get("id"), obj.get("cls"))

        lead_gap_b2b = None
        closing_speed_mps = None
        lead_speed_mps = None
        lead_id = None
        thw = float('inf')

        if closest is not None:
            y2, lead_id, lead_cls = closest

            # distance from camera center to ground point under the leader's rear (row y2)
            Z_cam_to_ground = self._ground_Z_from_y(y2)

            # estimate rear overhang (ground contact under rear axle is slightly ahead of the rear bumper)
            lead_rear_overhang = self.rear_overhang_by_cls.get(lead_cls, 0.9)

            # convert to ego front-bumper -> leader rear-bumper gap
            # (camera->ground) - (camera->ego-front) - (rear overhang)
            lead_gap_b2b = Z_cam_to_ground - self.cam_to_front_bumper_m - lead_rear_overhang
            # never let tiny negatives explode downstream
            lead_gap_b2b = max(lead_gap_b2b, 0.0)

            # --- closing speed on the GAP, not on camera-center distance ---
            current_time_s = float(t) / max(self.fps, 1)
            prev = self.vehicle_states.get(lead_id)
            if prev is not None:
                prev_t, prev_gap = prev
                dt = max(current_time_s - prev_t, 1.0 / max(self.fps, 1))
                # positive => approaching (gap shrinking)
                closing_speed_mps = max((prev_gap - lead_gap_b2b) / dt, 0.0)
                lead_speed_mps = ego_speed - closing_speed_mps
            # update state with the gap (not the raw Z)
            self.vehicle_states[lead_id] = (current_time_s, lead_gap_b2b)

            # THW from closing speed if we're actually closing; else infinite
            if closing_speed_mps and closing_speed_mps > self.SPEED_EPS:
                thw = lead_gap_b2b / closing_speed_mps

            # Console/debug
            print(f"Frame: {t}, Ego Speed: {ego_speed:.2f} m/s, "
                  f"Lead vehicle ID: {lead_id}, "
                  f"Lead Speed: {lead_speed_mps} m/s, "
                  f"Gap(b2b): {lead_gap_b2b:.2f} m, THW: {thw:.2f} s")

            # unsafe gates: (pixel proximity OR THW)
            vehicle_close_ahead = (y2 >= Y_THRESH)
            unsafe = vehicle_close_ahead or (thw < self.UNSAFE_THW_S)

        else:
            # no lead in path
            print(f"Frame: {t}, Ego Speed: {ego_speed:.2f} m/s, Lead Speed: {lead_speed_mps} m/s, Lead vehicle ID: None, Gap: N/A, THW: ∞")
            unsafe = False

        # latch/clear event
        if unsafe:
            if not self.unsafe_distance_active:
                self.unsafe_distance_active = True
                self.events_log.append({"event": "unsafe_follow_distance", "start": t, "end": None})
                self._output_warning(t, "unsafe_follow_distance")
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
        # Real-time output: console print and publish to voice assistant
        #print(message)
        if self.pub: # Check if pub socket exists
            try:
                # Assuming self.pub has a send_json method
                self.pub.send_json({"type": "warning", "frame": t, "description": description})
            except Exception as e:
                print(f"Error publishing warning: {e}", file=sys.stderr) # Log error if publishing fails

        # Log to file if enabled
        if self.log_file:
            try:
                log_message = f"{t},{description},{time.time()}"
                self.log_file.write(log_message + "\n")
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
