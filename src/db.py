"""
MongoDB connection singleton.
All modules that need database access import get_db() or get_gridfs() from here.
MongoClient is thread-safe and manages its own connection pool.
"""
import os
from pymongo import MongoClient
import gridfs

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        uri = os.environ["MONGODB_URI"]
        _client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    return _client


def get_db():
    return get_client()[os.getenv("MONGODB_DB_NAME", "aqi_db")]


def get_gridfs() -> gridfs.GridFS:
    return gridfs.GridFS(get_db())
