"""The JSON file helpers shared by the stores."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.storage.jsonfiles import PIECE_ITEMS, json_pieces, read_json, write_json

_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(10**6), 10**6),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=12),
)
_values = st.recursive(
    _scalars,
    lambda inner: st.one_of(
        st.lists(inner, max_size=7), st.dictionaries(st.text(max_size=6), inner, max_size=4)
    ),
    max_leaves=25,
)
_documents = st.dictionaries(st.text(max_size=8), _values, max_size=6)


@given(_documents, st.integers(1, 4))
@settings(max_examples=150, deadline=None)
def test_the_pieces_join_to_exactly_what_one_dumps_gives(data: dict, piece_items: int) -> None:
    """T-956: a long list is encoded a slice at a time; the file does not change by a byte."""
    assert "".join(json_pieces(data, piece_items=piece_items)) == json.dumps(
        data, indent=2, sort_keys=True
    )


def test_a_long_list_is_written_in_pieces_and_reads_back(tmp_path: Path) -> None:
    rows = [{"cause": f"a{i}", "effect": f"b{i}", "support": i} for i in range(PIECE_ITEMS * 2 + 7)]
    data = {"schema": 1, "relationships": rows, "variables": [], "title": "Game"}
    pieces = list(json_pieces(data))
    assert len(pieces) > 6  # the list came out in more than one piece
    path = write_json(tmp_path / "k.json", data, "knowledge")
    assert path.read_text("utf-8") == json.dumps(data, indent=2, sort_keys=True)
    assert read_json(path, "knowledge") == data


def test_data_that_is_not_json_leaves_no_half_file(tmp_path: Path) -> None:
    target = tmp_path / "k.json"
    with pytest.raises(TypeError):
        write_json(target, {"bad": {1, 2, 3}}, "knowledge")
    assert not target.exists() and not target.with_suffix(".json.tmp").exists()
    with pytest.raises(ConfigurationError):
        list(json_pieces({"a": 1}, piece_items=0))
