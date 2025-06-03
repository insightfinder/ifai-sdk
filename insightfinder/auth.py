class Auth:
    def __init__(self):
        self.username = None
        self.api_key = None

    def set_credentials(self, username, api_key):
        self.username = username
        self.api_key = api_key

    def get_credentials(self):
        return self.username, self.api_key

    def is_authenticated(self):
        return self.username is not None and self.api_key is not None