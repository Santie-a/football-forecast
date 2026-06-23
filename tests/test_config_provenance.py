"""Tests for RunConfig and the provenance manifest writer."""

import json

import pytest

from football_forecast.config import RunConfig
from football_forecast.provenance import file_fingerprint, write_manifest


def test_config_derives_intl_paths():
    c = RunConfig(source="intl")
    assert c.data.endswith("intl_results.parquet")
    assert c.raw.endswith("intl_results.csv")


def test_config_derives_league_paths():
    c = RunConfig(source="league", division="E0")
    assert c.data.endswith("league_E0.parquet")
    assert c.raw.endswith("club_matches.csv")


def test_config_rejects_bad_source():
    with pytest.raises(ValueError, match="source must be"):
        RunConfig(source="nope")


def test_config_roundtrip_toml(tmp_path):
    p = tmp_path / "run.toml"
    p.write_text('source = "league"\ndivision = "D1"\nmodel = "dixon_coles"\nseed = 7\n')
    c = RunConfig.from_toml(p)
    assert c.division == "D1" and c.seed == 7
    assert c.to_dict()["model"] == "dixon_coles"


def test_file_fingerprint(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    fp = file_fingerprint(f)
    assert fp["bytes"] == 5 and len(fp["sha256"]) == 16
    assert file_fingerprint(tmp_path / "missing") is None


def test_write_manifest(tmp_path):
    art = tmp_path / "model.pkl"
    art.write_text("x")
    inp = tmp_path / "in.parquet"
    inp.write_text("data")
    dest = write_manifest(
        art, stage="train", config={"model": "elo"}, seed=3,
        inputs=[inp], metrics={"rps": 0.18},
    )
    m = json.loads(dest.read_text())
    assert m["stage"] == "train" and m["seed"] == 3
    assert m["metrics"]["rps"] == 0.18
    assert m["inputs"][0]["path"].endswith("in.parquet")
    assert m["outputs"] == [str(art)]
    assert "created_utc" in m and "python" in m
