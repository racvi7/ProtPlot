# config.py

import os

class Config:
    """Base configuration."""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "your-default-secret-key")
    DATABASE_URI = os.environ.get("DATABASE_URI", "sqlite:///default.db")

class DevelopmentConfig(Config):
    """Development-specific configuration."""
    DEBUG = True

class TestingConfig(Config):
    """Testing-specific configuration."""
    TESTING = True
    DATABASE_URI = "sqlite:///:memory:"

class ProductionConfig(Config):
    """Production-specific configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY")  # Must be set
    DATABASE_URI = os.environ.get("DATABASE_URI")

# Optionally, config can be selected dynamically:
config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
