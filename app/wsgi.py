"""WSGI entry point for Gunicorn:  gunicorn app.wsgi:app"""
from app import create_app

app = create_app()
