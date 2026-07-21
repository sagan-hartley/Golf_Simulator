"""Tests for golf_simulator.domain."""

from golf_simulator.domain import TournamentType


def test_no_tournament_type_is_an_alias():
    # Two tournaments whose rules coincide (e.g. The Open and the PGA both use
    # a top-70-and-ties cut and a 156-player field) must remain distinct enum
    # members, not silently collapse into an alias of one another.
    aliases = [name for name, member in TournamentType.__members__.items() if member.name != name]
    assert aliases == []


def test_major_open_and_pga_are_distinct():
    assert TournamentType.MAJOR_OPEN is not TournamentType.MAJOR_PGA
    assert TournamentType.MAJOR_OPEN.name == "MAJOR_OPEN"
    assert TournamentType["MAJOR_OPEN"].name == "MAJOR_OPEN"


def test_every_tournament_has_a_distinct_label():
    labels = [t.value.label for t in TournamentType]
    assert len(labels) == len(set(labels))
