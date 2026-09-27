"""Shared test setup: hard-block real network access for every test."""

from __future__ import annotations

import socket

import pytest


def _no_network(*args: object, **kwargs: object) -> None:
    raise RuntimeError("Tests must not touch the network")


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to resolve a host or open a socket."""
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
