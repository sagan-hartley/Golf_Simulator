"""Tests for golf_simulator.season."""

import pytest

from golf_simulator.domain import TournamentType
from golf_simulator.season import validate_field_size


def _fake_player_params(n):
    return {f"Player{i}": (0.0, 70.0, 3.0, 1.0) for i in range(n)}


def test_validate_field_size_passes_when_large_enough():
    # PLAYOFF has the smallest field_size (70); REGULAR has the largest (156).
    player_params = _fake_player_params(156)
    validate_field_size(player_params, [TournamentType.REGULAR, TournamentType.PLAYOFF])


def test_validate_field_size_raises_when_too_small():
    player_params = _fake_player_params(50)
    with pytest.raises(ValueError, match="50.*156"):
        validate_field_size(player_params, [TournamentType.REGULAR])


def test_validate_field_size_empty_schedule_never_raises():
    player_params = _fake_player_params(1)
    validate_field_size(player_params, [])
