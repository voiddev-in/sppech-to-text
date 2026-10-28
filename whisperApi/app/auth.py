import hmac
from typing import Optional
from fastapi import Header, HTTPException, status
from app.config import settings


def constant_time_compare_key(provided_key: str, valid_keys: list[str]) -> bool:
    if not provided_key:
        return False
    matched = False
    for valid_key in valid_keys:
        if hmac.compare_digest(provided_key, valid_key):
            matched = True
    return matched


async def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Malformed authorization header. Expected 'Bearer <key>'.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )

    api_key = parts[1].strip()
    if not constant_time_compare_key(api_key, settings.api_keys_list):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Incorrect API key provided.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )

    return api_key
