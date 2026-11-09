"""WSGI entry point:  gunicorn api.wsgi:app"""
from api import create_api

app = create_api()
