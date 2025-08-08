import azure.cognitiveservices.speech as speechsdk
import zmq, time

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


def on_context_event(event):
    # Wrap text in SSML with cheerful style

    if event == "pedestrian_ahead":
        message = "Watch out for the pedestrian ahead"
    elif event == "Unsafe Following Distance":
        message = "Please try to maintain a safe distance behind the car ahead"

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
    result = speech_synthesizer.speak_ssml_async(ssml_template).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized for text [{message}] with cheerful style")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))


def main():
    zmq_context = zmq.Context()
    sub = zmq_context.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5575")  # Connect to context_engine publisher
    sub.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics (if any)
    print("Voice subscriber listening: ")
    try:
        while True:
            ctx_engine_msg = sub.recv_json()

            if ctx_engine_msg.get("type") != "warning":
                continue
            desc = ctx_engine_msg.get("description","Warning")
            on_context_event(desc)
    
    except KeyboardInterrupt:
        print("\nExiting. Closing ZeroMQ sockets and log file.")
        sub.close()
        zmq_context.term()

if __name__ == "__main__":
    main()

