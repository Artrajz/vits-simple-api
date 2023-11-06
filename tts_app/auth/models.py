class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)


def user2str(dict_data):
    users = {}
    for user_name, user_data in dict_data["users"]["admin"].items():
        users[user_name] = {'id': user_data.id, 'username': user_data.username, 'password': user_data.password}
    dict_data["users"]["admin"] = users
    return dict_data


def str2user(dict_data):
    users = {}
    for user_name, user_data in dict_data["users"]["admin"].items():
        users[user_name] = User(user_data['id'], user_data['username'], user_data['password'])
    dict_data["users"]["admin"] = users
    return dict_data

# users = {'group': {'username': User(1, 'username', 'password')}}
