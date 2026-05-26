"""Environment configuration for the Toxic Review Detector service.

The current application has a small configuration surface, but keeping settings
in this module avoids scattering environment lookups through API and model
code. Additional deployment controls can be added here as the platform grows.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base settings shared by every runtime environment."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    JSON_SORT_KEYS = False

class DevelopmentConfig(Config):
    """Local development settings with Flask debugging enabled."""
    DEBUG = True
    ENV = 'development'

class ProductionConfig(Config):
    """Production settings for hosted deployments."""
    DEBUG = False
    ENV = 'production'

class TestingConfig(Config):
    """Test settings for future automated API and utility tests."""
    TESTING = True
    DEBUG = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env=None):
    """Return the configuration class for the requested Flask environment."""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
