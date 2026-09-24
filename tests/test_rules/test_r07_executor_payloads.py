"""R07 — live/executor.py chip payloads (NO network: requests is stubbed).

WC/FH ride on POST /api/transfers/ ("wildcard" / "freehit");
BB/TC ride on POST /api/my-team/{id}/ ("bboost" / "3xc").
"""

from __future__ import annotations

import pytest

import fpl_optimizer.live.executor as ex

from tests.test_rules.rules_helpers import load_gt


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body
        self.text = ""

    def json(self):
        if self._body is None:
            raise ValueError("empty")
        return self._body


class _Auth:
    def headers(self):
        return {"Authorization": "Bearer test"}


@pytest.fixture
def posts(monkeypatch):
    sent = []

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append((url, json))
        return _Resp(200, {} if "transfers" in url else None)

    def forbid_get(*a, **k):
        raise AssertionError("no GETs expected")

    monkeypatch.setattr(ex.requests, "post", fake_post)
    monkeypatch.setattr(ex.requests, "get", forbid_get)
    return sent


ETYPES = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3,
          13: 4, 14: 4, 15: 4}
XI = [13, 8, 3, 1, 4, 9, 5, 10, 6, 11, 14]         # deliberately unsorted
BENCH = [12, 2, 7, 15]                             # GK not first on purpose
TR = [{"element_in": 16, "element_out": 7, "purchase_price": 45, "selling_price": 44}]


class TestTransferEndpointChips:
    @pytest.mark.parametrize("chip,api", [("wildcard", "wildcard"), ("free_hit", "freehit")])
    def test_wc_fh_names(self, posts, chip, api):
        ex.apply_transfers(_Auth(), 8737706, 11, TR, chip=chip, confirm=True)
        url, body = posts[-1]
        assert url.endswith("/api/transfers/")
        assert body["chip"] == api
        assert body["wildcard"] is (api == "wildcard")
        assert body["freehit"] is (api == "freehit")
        assert body["event"] == 11 and body["entry"] == 8737706
        assert body["confirmed"] is True

    def test_no_chip(self, posts):
        ex.apply_transfers(_Auth(), 8737706, 6, TR, chip=None, confirm=True)
        body = posts[-1][1]
        assert body["chip"] is None and body["wildcard"] is False and body["freehit"] is False

    @pytest.mark.parametrize("chip", ["bench_boost", "triple_captain", "bboost", "3xc"])
    def test_team_chips_rejected_on_transfer_endpoint(self, posts, chip):
        with pytest.raises(ValueError):
            ex.apply_transfers(_Auth(), 8737706, 6, TR, chip=chip)
        assert posts == []


class TestMyTeamEndpointChips:
    @pytest.mark.parametrize("chip,api", [("bench_boost", "bboost"),
                                          ("triple_captain", "3xc"), (None, None)])
    def test_bb_tc_names(self, posts, chip, api):
        ex.apply_lineup(_Auth(), 8737706, XI, BENCH, captain_id=13, vice_captain_id=8,
                        chip=chip, element_types=ETYPES)
        url, body = posts[-1]
        assert url.endswith("/api/my-team/8737706/")
        assert body["chip"] == api

    @pytest.mark.parametrize("chip", ["wildcard", "free_hit", "freehit"])
    def test_transfer_chips_rejected_on_my_team(self, posts, chip):
        with pytest.raises(ValueError):
            ex.apply_lineup(_Auth(), 8737706, XI, BENCH, 13, 8, chip=chip,
                            element_types=ETYPES)

    def test_pick_order_gk_pinned_at_12(self, posts):
        ex.apply_lineup(_Auth(), 8737706, XI, BENCH, 13, 8, element_types=ETYPES)
        picks = posts[-1][1]["picks"]
        assert [p["position"] for p in picks] == list(range(1, 16))
        types = [ETYPES[p["element"]] for p in picks]
        assert types[:11] == sorted(types[:11])         # GK, DEF.., MID.., FWD..
        assert picks[11]["element"] == 2                 # bench GK at 12
        assert [p["element"] for p in picks[12:]] == [12, 7, 15]  # outfield order kept
        caps = [p["element"] for p in picks if p["is_captain"]]
        vices = [p["element"] for p in picks if p["is_vice_captain"]]
        assert caps == [13] and vices == [8]

    def test_api_chip_vocabulary_matches_official(self):
        rules = load_gt("rules_2026-27.json")
        official = {c["name"] for c in rules["chips"]}
        used = set(ex._MYTEAM_CHIPS.values()) | set(ex._TRANSFER_CHIPS.values())
        assert used == official
        types = {c["name"]: c["chip_type"] for c in rules["chips"]}
        assert all(types[v] == "team" for v in ex._MYTEAM_CHIPS.values())
        assert all(types[v] == "transfer" for v in ex._TRANSFER_CHIPS.values())
