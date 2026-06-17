import os
import sys
from urllib.parse import unquote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app, ensure_user_tables

ensure_user_tables()

_wsgi = app.wsgi_app


# vercel's python runtime can hand the wsgi app a still-encoded PATH_INFO, so path
# variables like "/components/amd%202650" never match the route — decode before routing

def _decoded_path_app(environ, start_response):
    environ["PATH_INFO"] = unquote(environ.get("PATH_INFO", ""))
    return _wsgi(environ, start_response)


app.wsgi_app = _decoded_path_app
