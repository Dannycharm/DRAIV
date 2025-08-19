import azure.cognitiveservices.speech as speechsdk
import zmq, time, random

#speech_config = speechsdk.SpeechConfig(subscription="YOUR_AZURE_KEY", region="YOUR_REGION")
#speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"  # choose a natural voice
# audio_out = speechsdk.AudioOutputConfig(use_default_speaker=True)
#tts = speechsdk.SpeechSynthesizer(speech_config, audio_out)

'''
  For more samples please visit https://github.com/Azure-Samples/cognitive-services-speech-sdk
'''

# Creates an instance of a speech config with specified subscription key and service region.
speech_key = ""
service_region = "eastus2"


speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

# Voice name set here is overridden by SSML <voice> element, but keep for completeness
speech_config.speech_synthesis_voice_name = "en-US-Bree:DragonHDLatestNeural" #"en-US-Bree:DragonHDLatestNeural"
#audio_config = speechsdk.audio.AudioConfig(filename="intro2.wav")
# speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)



def on_context_event(event, obj_class, first=False):
    # Wrap text in SSML with cheerful style
    #first = True keeps the message short and subsequent messages are more detailed

    if first:
        unsafe_distance_messages = [f"Please maintain a safe distance behind the {obj_class} ahead"]
        safe_distance_messages = [ "Thank's for maintaining a safe distance",
                                   "Following distance is now safe",
                                   "Safe distance restored"]
    else:
        unsafe_distance_messages = [f"Please maintain a safe distance behind the {obj_class} ahead", 
                                    f"You're following too closely behind the {obj_class} ahead, please increase your distance",
                                    f"Please slow down and increase your distance from the {obj_class} ahead"]
        safe_distance_messages = [
                                    f"You're keeping a safe distance from the {obj_class} ahead, great job!",
                                    f"Safe following distance maintained. This helps improve reaction time.",
                                    f"You're at a safe distance --- this reduces the risk of collisions.",
                                    f"Maintaining this space gives you more time to react."
                                ]

    audio_duration_seconds = 0

    if event == "pedestrian_ahead":
        message = "Watch out for the pedestrian ahead"
    elif event == "unsafe_follow_distance":
        message = random.choice(unsafe_distance_messages)
    elif event == "follow_distance_safe":
        message = random.choice(safe_distance_messages)
    else: 
        message = "Warning"
    
    ssml_template = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
           xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
      <voice name="en-US-Bree:DragonHDLatestNeural">
        <mstts:express-as style="cheerful">
          {message}
        </mstts:express-as>
      </voice>
    </speak>
    """
    start_time = time.time()
    result = speech_synthesizer.speak_ssml_async(ssml_template).get()
    end_time = time.time()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized for text [{message}] with cheerful style")
        audio_duration_seconds = result.audio_duration.total_seconds()
        print(f"Audio duration: {audio_duration_seconds:.2f} seconds")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
    return audio_duration_seconds, end_time, (end_time - start_time)


def main():
    print("The time is:", time.time())
    zmq_context = zmq.Context()
    sub = zmq_context.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5575")  # Connect to context_engine publisher
    sub.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics (if any)
    # Set a timeout for the receive operation (e.g., 100 milliseconds)
    sub.setsockopt(zmq.RCVTIMEO, 100) 
    
    log_version = "two"  # Change this to match your sample version"
    
    event_log = []
    event_states = {} # {message:(count,last_time_seen)}
    non_urgent_threshold = 300.0 # seconds 
    warning_state = ""
    print("Voice subscriber listening: ")
    
    try:
        while True:
            try:
                ctx_engine_msg = sub.recv_json()
            except zmq.Again:
                continue

            if ctx_engine_msg.get("type") != "warning":
                continue
            
            desc = ctx_engine_msg.get("description")
            if desc not in event_states:
                event_states[desc] = (1, time.time())
            
            frame = ctx_engine_msg.get("frame")
            seen_time = ctx_engine_msg.get("seen_time")
            obj_class = ctx_engine_msg.get("obj_cls", "vehicle")

            if event_states[desc][0] == 1:
                audio_duration,end_time,cloud_latency = on_context_event(desc, obj_class, first=True)
                event_states[desc] = (event_states[desc][0]+1, time.time())
                warning_state = "new"

            elif event_states[desc][0] > 1 and (time.time() - event_states[desc][1]) > non_urgent_threshold:           
                audio_duration,end_time,cloud_latency = on_context_event(desc, obj_class, first=True)
                event_states[desc] = (event_states[desc][0]+1, time.time())
                warning_state = "new"
            else:
                audio_duration,end_time,cloud_latency = on_context_event(desc, obj_class, first=False)
                event_states[desc] = (event_states[desc][0]+1, time.time())
                warning_state = "repeat"

            if seen_time and end_time:
                overall_latency = end_time - seen_time
                start_latency = overall_latency - audio_duration
                margin_to_2s = 1 - start_latency
                audio_start_time = end_time - audio_duration
                event_log.append((frame, audio_start_time, audio_duration, seen_time, end_time, overall_latency, cloud_latency, start_latency, margin_to_2s, warning_state))
    except KeyboardInterrupt:
        print("\nExiting. Closing ZeroMQ sockets and log file.")
    
    # The `finally` block should be here, outside the while loop,
    # to perform cleanup only when the loop is exited.
    finally:
        with open(f"voice_assistant_{log_version}.log", "w") as file:
            file.write("frame,audio_start_time,audio_duration,seen_time,end_time,overall_latency,cloud_latency,start_latency,margin_to_2s,warning_state\n")
            for f, a_s_t, a_d, s_t, e_t, o_l, c_l, s_l, m_t_2, w_s in event_log:
                file.write(f"{f},{a_s_t:.2f},{a_d:.2f},{s_t:.2f},{e_t:.2f},{o_l:.2f},{c_l:.2f},{s_l:.2f},{m_t_2:.2f},{w_s}\n")
        print("Voice assistant log saved")
        sub.close()
        zmq_context.term()


if __name__ == "__main__":
    main()


