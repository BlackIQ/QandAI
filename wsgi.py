from app.main import app

from app.config.config import env


if __name__ == "__main__":
    app.run(port=env["APP_PORT"], debug=True)
