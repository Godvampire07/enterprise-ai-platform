import redis
from backend.app.core.config import settings
from backend.app.core.logging import logger

class RedisClient:
    def __init__(self):
        self.client = redis.from_url(
            str(settings.REDIS_URL),
            decode_responses=True,
            socket_timeout=5
        )

    def set(self, key: str, value: str, expire: int = None):
        try:
            self.client.set(key, value, ex=expire)
        except redis.RedisError as e:
            logger.error(f"Redis set error: {e}")

    def get(self, key: str) -> str:
        try:
            return self.client.get(key)
        except redis.RedisError as e:
            logger.error(f"Redis get error: {e}")
            return None

    def delete(self, key: str):
        try:
            self.client.delete(key)
        except redis.RedisError as e:
            logger.error(f"Redis delete error: {e}")

redis_client = RedisClient()

def get_redis():
    return redis_client
