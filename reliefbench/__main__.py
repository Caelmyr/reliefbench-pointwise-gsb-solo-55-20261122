import os

from waitress import serve

from .http import create_app


def main():
    app = create_app()
    serve(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")),
          threads=4, channel_timeout=45, max_request_body_size=80 * 1024 * 1024)


if __name__ == "__main__":
    main()
