"""Run with the built-in Flask server:  python -m app

Debug mode is OFF unless FLASK_DEBUG=1: the debugger lets anyone who reaches
the page execute arbitrary Python, the reloader loads the model twice, and
tracebacks leak internals. In containers Gunicorn (a production WSGI server)
is used instead of this development server.
"""
import os

from app import create_app

if __name__ == "__main__":
    app = create_app()
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")),
            debug=app.config["DEBUG"], use_reloader=False)
