from .. import app

import logging
import os


@app.route('/')
def default():
    # logging.info('This is an info message')
    # logging.warning('This is a warning message')
    # logging.error('This is an error message')
    # logging.critical('This is a critical message')

    flask_env = os.environ.get("FLASK_ENV", "development")
    return f'Template backend is up! : {flask_env}'