"""Validated evidence identities and content fingerprints, not trust judgments."""

import hashlib
import json
from urllib.parse import urlparse

from pydantic import Field, field_validator

from .models import StrictModel


class EvidenceDocument(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    title: str = Field(min_length=1, max_length=300)
    source_url: str = Field(min_length=1, max_length=2000)
    text: str = Field(min_length=1, max_length=12000)

    @field_validator("source_url")
    @classmethod
    def safe_reference(cls, value):
        url = urlparse(value)
        if url.scheme == "https" and url.hostname and not url.username and not url.password:
            return value
        if (
            url.scheme == "local"
            and url.netloc == "docs"
            and url.path.startswith("/")
            and ".." not in url.path.split("/")
        ):
            return value
        raise ValueError("source must be an HTTPS reference or a local://docs/ reference")


def content_digest(value: dict | list) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
