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


class BartenderBot:
    def __init__(self, furhat):
        self.furhat = furhat
        self.conversation_state = {}
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

        response = {
            "greeting": behaviors.get("greeting", ""),
            "drink_suggestion": f"How about our special '{behaviors.get('drink_suggestion')}' to match your mood?",
            "conversation": speech_response,
            "farewell": behaviors.get("farewell", ""),
        }
        return response

    def on_user_interaction(self, user_emotion, user_speech):
        response = self.generate_bot_response(user_emotion, user_speech)

        if response["greeting"]:
            self.furhat.say(response["greeting"])
        if response["drink_suggestion"]:
            self.furhat.say(response["drink_suggestion"])
        if response["conversation"]:
            self.furhat.say(response["conversation"])
        if response["farewell"]:
            self.furhat.say(response["farewell"])

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


# # Initialize Furhat and BartenderBot
# furhat_robot = Furhat("localhost")
# furhat_robot.set_voice(name="Matthew")
# furhat_robot.set_led(red=200, green=50, blue=50)

# bot = BartenderBot(furhat_robot)

# user_emotion = "happy"  # This would come from your emotion detection subsystem
# user_speech = "It's my birthday today!"  # This would come from speech recognition

# bot.on_user_interaction(user_emotion, user_speech)
