import time
import threading
from flask import request, jsonify
from functools import wraps

class TokenBucket:
    def __init__(self, capacity, fill_rate):
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = capacity
        self.last_fill = time.time()
        self.lock = threading.Lock() # Ensures thread safety during parallel requests

    def consume(self, tokens=1):
        with self.lock:
            now = time.time()
            # Add new tokens based on time passed
            self.tokens += (now - self.last_fill) * self.fill_rate
            if self.tokens > self.capacity:
                self.tokens = self.capacity
            self.last_fill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

# Global bucket: 5 requests per second max
bucket = TokenBucket(capacity=5, fill_rate=1)

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not bucket.consume():
            return jsonify({"error": "429 Too Many Requests - Rate limit exceeded"}), 429
        return f(*args, **kwargs)
    return decorated_function
