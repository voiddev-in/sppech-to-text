import asyncio
from contextlib import asynccontextmanager
from fastapi import HTTPException, status
from app.config import settings


class ConcurrencyLimiter:
    def __init__(self, max_concurrency: int, max_queue: int):
        self.max_concurrency = max(1, max_concurrency)
        self.max_queue = max(0, max_queue)
        self._active = 0
        self._queued = 0
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._lock = asyncio.Lock()

    @property
    def queue_depth(self) -> int:
        return self._queued

    @property
    def active_workers(self) -> int:
        return self._active

    @asynccontextmanager
    async def acquire_slot(self):
        async with self._lock:
            if self._active + self._queued >= self.max_concurrency + self.max_queue:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": {
                            "message": "Server is busy. Queue is full, please retry your request shortly.",
                            "type": "server_error",
                            "param": None,
                            "code": "service_unavailable",
                        }
                    },
                )
            self._queued += 1

        try:
            await self._semaphore.acquire()
            async with self._lock:
                self._queued -= 1
                self._active += 1
            try:
                yield
            finally:
                async with self._lock:
                    self._active -= 1
                self._semaphore.release()
        except BaseException:
            # If an exception happened before acquiring semaphore (e.g., client disconnected / task cancelled)
            if self._queued > 0 and self._semaphore.locked():
                async with self._lock:
                    self._queued = max(0, self._queued - 1)
            raise


limiter = ConcurrencyLimiter(
    max_concurrency=settings.whisper_max_concurrency,
    max_queue=settings.whisper_max_queue,
)
