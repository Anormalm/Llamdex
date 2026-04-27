from src.multimodal.utils import backbone


def test_exists_in_hf_cache_does_not_false_positive_on_listdir_error(monkeypatch, tmp_path):
    cache_root = tmp_path / "hf_cache"
    (cache_root / "models--foo--bar" / "snapshots").mkdir(parents=True)
    monkeypatch.setattr(backbone.os.path, "expanduser", lambda _: str(tmp_path / "missing_home_cache"))

    def _raise_permission_error(_):
        raise PermissionError("denied")

    monkeypatch.setattr(backbone.os, "listdir", _raise_permission_error)
    assert backbone._exists_in_hf_cache(str(cache_root), "foo/bar") is False
