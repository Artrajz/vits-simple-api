from functools import wraps
from flask import request, jsonify, make_response
from config import config


def require_api_key(func):
    @wraps(func)
    def check_api_key(*args, **kwargs):
        if not config.system.api_key_enabled:
            return func(*args, **kwargs)
        else:
            api_key = request.args.get('api_key') or request.headers.get('X-API-KEY')
            if api_key and api_key == config.system.api_key:
                return func(*args, **kwargs)
            else:
                return make_response(jsonify({"status": "error", "message": "Invalid API Key"}), 401)

    return check_api_key
