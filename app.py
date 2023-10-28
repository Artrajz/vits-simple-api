from flask import Flask

from tts_app import init_routes

app = Flask(__name__)
app.config.from_pyfile("config.py")

# Initialize routes
init_routes(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config.get("PORT", 23456), debug=app.config.get("DEBUG", False))
