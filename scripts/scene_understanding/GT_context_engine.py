import sys
import time

class ContextEngine:
    def __init__(self, log_to_file=False, log_file_path=None, pub=None):
        """
        Initialize the context engine.
        :param log_to_file: If True, enable logging of events to a file.
        :param log_file_path: Path to the log file (used if log_to_file is True).
        :param pub: An optional publisher object (e.g., a ZeroMQ socket) to send warnings.
        """
        self.log_to_file = log_to_file
        self.log_file = None
        if self.log_to_file and log_file_path:
            try:
                self.log_file = open(log_file_path, 'w')
                self.log_file.write("frame,description,sent_time\n")
            except IOError as e:
                print(f"Error opening log file {log_file_path}: {e}", file=sys.stderr)
                self.log_to_file = False

        # --- Threshold Parameters ---
        self.UNSAFE_THW_S = 3.0
        self.fps = 30.0
        
        # --- Time-based Alert Parameters ---
        self.WARN_DURATION_S = 0.5  # Must be unsafe for this long to trigger a warning
        self.SAFE_DURATION_S = 2.0  # Must be safe for this long to send a 'safe' message

        # --- State Management ---
        # "SAFE": Normal state.
        # "MONITORING_UNSAFE": Unsafe condition detected, timer started.
        # "WARNING_SENT": Warning has been sent, now monitoring for safe condition.
        # "MONITORING_SAFE": Safe condition detected after a warning, timer started.
        self.follow_state = "SAFE"
        self.unsafe_start_time = 0.0
        self.safe_start_time = 0.0

        # Log of past events
        self.events_log = []

        # To publish events
        self.pub = pub

    def process_frame(self, msg):
        """
        Process a single perception output frame, managing state and time-based alerts.
        """
        t_frame = msg.get("t")
        current_time_s = t_frame / self.fps
        vehicle_types = {"car","truck","bus","motorcycle","bicycle","rider","trailer","other vehicle"}        
        objects = msg.get("objects") or []
        obj_class = None
        thw = msg.get("thw")
        seen_time =  msg.get("seen_time")
        is_unsafe = thw < self.UNSAFE_THW_S

        # --- State Machine for Following Distance ---

        if self.follow_state == "SAFE":
            if is_unsafe:
                # Transition to monitoring when an unsafe condition starts
                self.follow_state = "MONITORING_UNSAFE"
                self.unsafe_start_time = current_time_s
        
        elif self.follow_state == "MONITORING_UNSAFE":
            if is_unsafe:
                # Check if the unsafe duration has passed the threshold
                duration = current_time_s - self.unsafe_start_time
                if duration >= self.WARN_DURATION_S:
                    self._send_event_message(t_frame, "unsafe_follow_distance", obj_class, seen_time)
                    self.events_log.append({"event": "unsafe_follow_distance", "start": t_frame, "end": None})
                    self.follow_state = "WARNING_SENT"
            else:
                # If condition resolves before warning, go back to safe
                self.follow_state = "SAFE"

        elif self.follow_state == "WARNING_SENT":
            if not is_unsafe:
                # Condition is now safe, start monitoring for stable safety
                self.follow_state = "MONITORING_SAFE"
                self.safe_start_time = current_time_s

        elif self.follow_state == "MONITORING_SAFE":
            if not is_unsafe:
                # Check if the safe duration has passed the threshold
                duration = current_time_s - self.safe_start_time
                if duration >= self.SAFE_DURATION_S:
                    self._send_event_message(t_frame, "follow_distance_safe", seen_time=seen_time)
                    # Find the corresponding unsafe event and mark its end
                    for ev in reversed(self.events_log):
                        if ev["event"] == "unsafe_follow_distance" and ev["end"] is None:
                            ev["end"] = t_frame
                            break
                    self.follow_state = "SAFE" # Reset to the initial safe state
            else:
                # If it becomes unsafe again before the safe period is confirmed,
                # revert to the warning state (no new warning is sent).
                self.follow_state = "WARNING_SENT"

    def _send_event_message(self, t, description, obj_class=None, seen_time=None):
        """
        Helper to output a message (print, publish, and log to file).
        """
        # Publish to voice assistant
        if self.pub:
            try:
                payload = {"type": "warning", "frame": t, "description": description, "seen_time": seen_time}
                if obj_class is not None:
                    payload["obj_class"] = obj_class
                self.pub.send_json(payload)
            except Exception as e:
                print(f"Error publishing message: {e}", file=sys.stderr)

        # Log to file if enabled
        if self.log_file:
            try:
                log_message = f"{t},{description},{time.time()}"
                self.log_file.write(log_message + "\n")
                self.log_file.flush()
            except IOError as e:
                print(f"Error writing to log file: {e}", file=sys.stderr)
                self.log_to_file = False

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
