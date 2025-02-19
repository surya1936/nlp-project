from sklearn import pipeline
import speech_recognition as sr
import pyttsx3
from process.api.ai import openai
#import threading
#import time
from transformers import pipeline

 # For GPT-like AI responses
import process.api.gpu as gpu
# Initialize AI Model (Hugging Face Pipeline)
print("Loading AI model...")
ai_pipeline = pipeline("text-generation", model="gpt2", device=0)  # Use GPU if available

recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty("voices")
#for idx, voice in enumerate(voices): print(f"Voice {idx}: {voice.name}") # male voice 
for voice in voices: 
    if "male" in voice.name.lower(): 
        tts_engine.setProperty('voice', voice.id) 
        break
#tts_engine.setProperty("voice", voices[1].id) #female voice
tts_engine.setProperty("rate", 180) 
tts_engine.setProperty("volume", 1.0)  


def generate_ai_response(user_input):
    if "stop" in user_input.lower():
        return "Goodbye! Stopping the conversation."
    
    try: # Use OpenAI's GPT-3.5 to generate a response 
        response = openai.Completion.create( engine="gpt-3.5-turbo", 
                                            prompt=user_input, 
                                            max_tokens=150 ) 
        return response.choices[0].text.strip() 
    except Exception as e: 
        print(f"Error with OpenAI API: {e}")
        
    response = ai_pipeline(user_input, max_length=50, num_return_sequences=1)[0]["generated_text"]
    return response.strip()

#  Speak AI Response
def speak_response(response):
    tts_engine.say(response)
    tts_engine.runAndWait()

def real_time_conversation():
    print("Starting real-time AI conversation... (say 'stop' to end)")

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Handle background noise

        try:
            while True:
                print("Listening for user input...")
                
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
               
               
                try:
                    user_input = recognizer.recognize_google(audio)
                    print(f"User said: {user_input}")
                    
                    ai_response = generate_ai_response(user_input)
                    print(f"AI: {ai_response}")
                    speak_response(ai_response)

                    if "stop" in user_input.lower():
                        print("Stopping the conversation.")
                        break

                except sr.UnknownValueError:
                    print("Sorry, I couldn't understand that. Please speak clearly.")
                except sr.RequestError as e:
                    print(f"Google Speech Recognition error: {e}")

        except KeyboardInterrupt:
            print("Conversation ended by user.")
        finally:
            print("Goodbye!")

if __name__ == "__main__":
    real_time_conversation()



