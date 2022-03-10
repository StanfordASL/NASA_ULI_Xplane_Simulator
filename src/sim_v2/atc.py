class ATCAgent:
    def __init__(self, client):
        self.client=client
        self.message=""
    
    def speak(self, ac=0):
        self.message="" #TODO