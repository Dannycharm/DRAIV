# zure SDK for a proactive TTS announcement
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
# Note: the voice setting will not overwrite the voice element in input SSML.
speech_config.speech_synthesis_voice_name = "en-US-BrandonMultilingualNeural"
# speech_config.speech_synthesis_voice_name = "Ava Multilingual"

# use the default speaker as audio output.
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# This function is called whenever a context JSON event is emitted:
def on_context_event(event_json):
    #if event_json.get("event") == "pedestrian_ahead":
        # Compose the announcement
        message = "Watch out for the pedestrian ahead"
        result = speech_synthesizer.speak_text_async(message).get()
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized for text [{}]".format(message))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

def main():
    #zmq_context = zmq.Context() 
    #sub = zmq_context.socket(zmq.SUB)
    #sub.connect("tcp://*:5560")  # Connect to context_engine publisher
    #sub.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics (if any)
    #ctx_engine_msg = sub.recv_json()
    # Test
    ctx_engine_msg = "Please watch out for the pedestrian ahead, thank you!" 
    
    on_context_event(ctx_engine_msg)

if __name__ == "__main__":
    main()
