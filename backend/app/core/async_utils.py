"""Async helpers that remain compatible with Python 3.8."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Callable, TypeVar

T = TypeVar("T")


async def run_blocking(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking callable in the default executor.

    This is the Python 3.8 equivalent of ``asyncio.to_thread()``, which was
    introduced in Python 3.9.
    """

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))
