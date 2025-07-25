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
service_region = "eastus2"

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

# Voice name set here is overridden by SSML <voice> element, but keep for completeness
speech_config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"

speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)


def on_context_event(event_text):
    # Wrap text in SSML with cheerful style
    ssml_template = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
           xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
      <voice name="en-US-AvaMultilingualNeural">
        <mstts:express-as style="cheerful">
          {event_text}
        </mstts:express-as>
      </voice>
    </speak>
    """
    result = speech_synthesizer.speak_ssml_async(ssml_template).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized for text [{event_text}] with cheerful style")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))


def main():
    test_message = "Please watch out for the pedestrian ahead, thank you!"
    on_context_event(test_message)


if __name__ == "__main__":
    main()
