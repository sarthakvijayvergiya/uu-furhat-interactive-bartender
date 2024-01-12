from furhat_remote_api import FurhatRemoteAPI


class Furhat:
    def __init__(self, api_endpoint):
        self.api = FurhatRemoteAPI(api_endpoint)

    def say(self, text):
        self.api.say(text=text, blocking=True)

    def gesture(self, name):
        self.api.gesture(name=name, blocking=True)

    def set_led(self, red, green, blue):
        self.api.set_led(red=red, green=green, blue=blue)

    def set_voice(self, name):
        self.api.set_voice(name=name)
    
    def listen(self):
        self.api.listen()


class BartenderBot:
    def __init__(self, furhat):
        self.furhat = furhat
        self.conversation_state = {}
        self.waiting_for_user_response = False
        # Start listening for user speech
        self.listen_for_user_speech()
        self.emotion_behavior_map = {
            "happy": {
                "greeting": "It looks like you're in a great mood today! Let's keep that going.",
                "drink_suggestions": [
                    "Sunshine Cocktail",
                    "Joyful Julep",
                    "Blissful Bellini",
                ],
                "conversation": [
                    "What brings you here today?",
                    "Got any special celebration?",
                ],
                "farewell": "Glad I could add to your happy day! Come back anytime!",
            },
            "sad": {
                "greeting": "Seems like you could use a pick-me-up.",
                "drink_suggestions": [
                    "Comfort Cappuccino",
                    "Soothing Smoothie",
                    "Consolation Cooler",
                ],
                "conversation": [
                    "I'm here if you want to talk.",
                    "Hope this drink cheers you up.",
                ],
                "farewell": "Hope you feel better soon. Take care!",
            }
        }
        self.gesture_map = {
            "greeting": "Wink",  
            "farewell": "Goodbye",  
            "happy": "Smile", 
            "sad": "BigSmile",  
        }
        self.initialize_conversation()

    def handle_keywords(self, user_speech):
        if "birthday" in user_speech.lower():
            self.furhat.say("Happy Birthday! Let me make this day special for you.")
        elif "work" in user_speech.lower():
            self.furhat.say(
                "Work can be quite the adventure. What do you do for a living?"
            )

    def generate_bot_response(self, user_emotion, user_speech):
        behaviors = self.emotion_behavior_map.get(user_emotion, {})

        speech_response = self.map_speech_to_behavior(user_speech)
        gesture = self.gesture_map.get(user_emotion)
        response = {
            "greeting": behaviors.get("greeting", ""),
            "drink_suggestion": f"How about our special '{behaviors.get('drink_suggestion')}' to match your mood?",
            "conversation": speech_response,
            "farewell": behaviors.get("farewell", ""),
            "gesture": gesture
        }
        return response

    def initialize_conversation(self):
        # Initialize the conversation state or any other setup needed
        self.conversation_state = {}

        # Welcome the user and provide an introduction
        self.furhat.say("Welcome to the Cocktail AI project!")
        self.furhat.say("I'm here to assist you with drink suggestions and more.")

        # Start a conversation with a question
        self.furhat.say("What brings you here today? Are you looking for a special drink or just exploring?")
        self.waiting_for_user_response = True

    def is_waiting_for_user_response(self):
        # Method to check the state of waiting_for_speech
        return self.waiting_for_user_response

    def listen_for_user_speech(self):
    # Continuous loop for listening to user speech
        while True:
            while self.waiting_for_user_response:
                # Simulate Furhat listening for the user's speech
                # In a real scenario, this would involve interacting with Furhat's speech recognition API
                # For now, we'll simulate by waiting for user input
                resp = self.furhat.listen()
                if resp:
                    user_speech = resp.message
                    self.update_conversation_state(user_speech)
                    self.update_conversation_state(user_speech)
    
    def on_user_interaction(self, user_emotion):
        while self.waiting_for_user_response:
            user_speech = self.conversation_state.get("last_speech", "")
            response = self.generate_bot_response(user_emotion, user_speech)

            if response["greeting"]:
                self.furhat.say(response["greeting"], gesture=self.gesture_map.get("greeting"))
            if response["drink_suggestion"]:
                self.furhat.say(response["drink_suggestion"])
            if response["conversation"]:
                self.furhat.say(response["conversation"])
            if response["farewell"]:
                self.furhat.say(response["farewell"], gesture=self.gesture_map.get("farewell"))

    def serve_drink(self, drink_name):
        self.furhat.say(f"Here's your {drink_name}. Enjoy!")

    def map_speech_to_behavior(self, user_speech):
        if "?" in user_speech:
            return "That's an interesting question."
        elif "thanks" in user_speech.lower():
            return "You're welcome! Anything else I can do for you?"
        else:
            return "That's nice to hear."

    def update_conversation_state(self, user_speech):
        # Update the conversation state based on user speech
        # This can be as simple or complex as needed
        self.conversation_state["last_speech"] = user_speech
