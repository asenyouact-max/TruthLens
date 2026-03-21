import json

class MockRedis:
    def __init__(self):
        self.data = {}
    async def setex(self, key, ttl, value):
        self.data[key] = value
    async def get(self, key):
        return self.data.get(key)

_redis_client = MockRedis()

async def init_redis():
    pass

async def get_redis():
    return _redis_client

async def set_session_state(session_id: str, state: dict, ttl: int = 3600):
    await _redis_client.setex(f"session:{session_id}", ttl, json.dumps(state))

async def get_session_state(session_id: str) -> dict:
    data = await _redis_client.get(f"session:{session_id}")
    return json.loads(data) if data else None

async def update_session_state(session_id: str, updates: dict):
    state = await get_session_state(session_id) or {}
    state.update(updates)
    await set_session_state(session_id, state)
