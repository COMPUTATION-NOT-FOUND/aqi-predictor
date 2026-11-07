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


# Days to keep each hourly collection. MongoDB deletes older documents on its own
# (a TTL index), so the free 512 MB Atlas tier never fills up. aqi_features is
# not listed: it is the training data and stays small (about 10 MB a year).
TTL_DAYS = {
    ("predictions", "predicted_at"):  7,    # the dashboard only reads the latest
    ("prediction_log", "logged_at"):  90,   # track-record chart
    ("cams_forecasts", "issued_at"):  120,  # CAMS baseline scoring
}


def ensure_ttl_indexes() -> None:
    """Create the TTL indexes once. Safe to call on every run."""
    db = get_db()
    for (col, field), days in TTL_DAYS.items():
        try:
            db[col].create_index(field, expireAfterSeconds=days * 86400,
                                 name=f"ttl_{field}")
        except Exception as e:  # an older non-TTL index on the same field
            print(f"[db] TTL index on {col}.{field} not created: {e}")


def storage_stats() -> dict:
    """Database size in MB, for the dashboard footer."""
    s = get_db().command("dbStats", scale=1024 * 1024)
    return {"data_mb": s.get("dataSize", 0), "storage_mb": s.get("storageSize", 0),
            "index_mb": s.get("indexSize", 0), "objects": s.get("objects", 0)}
