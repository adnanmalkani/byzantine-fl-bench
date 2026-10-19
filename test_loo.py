"""Checks for the leave-out controls (Config.exclude_ids, 2026-10-18).

1. The default path is unchanged: stored grid results (heart, TCGA, ISIC)
   are reproduced bit for bit (history and diagnostics).
2. Exact twins: an excluded client still draws its batches, so in round 1
   every remaining client's update is identical in the control, the clean
   run with everyone, and the attacked run (round 1 starts from the same
   model, so the updates depend only on the batches).
3. Validity: the count check is skipped only for rules that ignore f.
4. All n test sets are evaluated, the excluded client's included.
"""
import json
import re
from pathlib import Path

from bench.fl import make_config, run
from run import run_name

STORED = [  # (folder, run name) from the main grids; heart files keep the pre-lr name format
    ("results_heart", "fed_heart_disease_a0.5_cc_w_tau0.03_sign_flip_byz0_ls100_r106_s3"),
    ("results_heart", "fed_heart_disease_a0.5_median_none_b0.0_fa1_ls100_r106_s0"),
    ("results_heart", "fed_heart_disease_a0.5_geomed_gaussian_byz1_ls100_r106_s7"),
    ("results_tcga", "fed_tcga_brca_a0.5_trimmed_mean_ipm_byz0_ls100_lr0.003_r243_s0"),
    ("results_tcga", "fed_tcga_brca_a0.5_cc_w_tau0.03_none_b0.0_fa2_ls100_lr0.003_r243_s1"),
    ("results_isic", "fed_isic2019_a0.5_multi_krum_label_flip_byz2_ls100_lr0.01_r101_s5"),
    ("results_isic", "fed_isic2019_a0.5_median_w_alie_byz0-1_ls100_lr0.01_r101_s0"),
    ("results_isic", "fed_isic2019_a0.5_krum_none_b0.0_fa0_ls100_lr0.01_r101_s0"),
]


def _cfg(stored):
    c = dict(stored["config"])
    return make_config(**c)


def test_default_path_reproduces_stored_results():
    checked = 0
    for folder, name in STORED:
        p = Path(folder) / f"{name}.json"
        if not p.exists():   # result_path() renames old-format files (no _lr) on lookup
            p = next(Path(folder).glob(re.sub(r"_r(\d+)_s", r"_lr*_r\1_s", name) + ".json"), p)
        if not p.exists():
            print(f"skipped (missing): {p}")
            continue
        old = json.loads(p.read_text())
        cfg = _cfg(old)
        assert p.stem in (run_name(cfg), run_name(cfg, lr=False)), (run_name(cfg), p.stem)
        new = run(cfg, log=lambda *_: None)
        assert new["history"] == old["history"], name
        assert new["diag"] == old["diag"], name
        assert new["byz_ids"] == old["byz_ids"] and "excluded_ids" not in new
        checked += 1
    print(f"{checked} stored runs reproduced bit for bit")


def test_exact_twins_and_test_sets():
    kw = dict(dataset="fed_heart_disease", aggregator="median", rounds=2, f_assumed=1, seed=4)
    full = run(make_config(**kw), log=lambda *_: None)
    ctrl = run(make_config(**kw, exclude_ids="1"), log=lambda *_: None)
    att = run(make_config(**{**kw, "attack": "sign_flip", "byz_ids": "1"}), log=lambda *_: None)
    d_full, d_ctrl, d_att = (r["diag"][0]["dist_prev"] for r in (full, ctrl, att))
    for i in (0, 2, 3):   # round 1: ||u_i||, depends only on client i's batches
        assert d_ctrl[i] == d_full[i] == d_att[i], i
    assert d_ctrl[1] is None and ctrl["excluded_ids"] == [1]
    assert len(ctrl["history"][-1]["client_acc"]) == 4     # all 4 test sets, incl. client 1's
    assert "_ex1_" in run_name(make_config(**kw, exclude_ids="1"))
    # ISIC, weighted, a pair left out: same twin property
    kw = dict(dataset="fed_isic2019", aggregator="cc", weighted=True, rounds=1,
              local_steps=5, f_assumed=2)
    full = run(make_config(**kw), log=lambda *_: None)
    ctrl = run(make_config(**kw, exclude_ids="0,3"), log=lambda *_: None)
    assert ctrl["status"] == "ok" and len(ctrl["history"][-1]["client_acc"]) == 6
    for i in (1, 2, 4, 5):
        assert ctrl["diag"][0]["dist_prev"][i] == full["diag"][0]["dist_prev"][i], i


def test_validity_in_controls():
    # n - 2 = 4 clients, f_assumed = 2: rules that ignore f run; rules using f don't
    for agg, ok in (("median", True), ("geomed", True), ("cc", True), ("fedavg", True),
                    ("trimmed_mean", False), ("krum", False)):
        r = run(make_config(dataset="fed_tcga_brca", aggregator=agg, rounds=1, local_steps=2,
                            f_assumed=2, exclude_ids="0,1"), log=lambda *_: None)
        assert (r["status"] == "ok") == ok, (agg, r.get("reason"))
    # n - 1 = 3 heart clients, f_assumed = 1: Krum invalid as in the attacked grid
    r = run(make_config(dataset="fed_heart_disease", aggregator="krum", rounds=1,
                        f_assumed=1, exclude_ids="2"), log=lambda *_: None)
    assert r["status"] == "invalid"
    # controls are clean runs only
    try:
        run(make_config(dataset="fed_heart_disease", attack="sign_flip", byz_ids="0",
                        exclude_ids="1", rounds=1), log=lambda *_: None)
        raise RuntimeError("attack accepted in a control")
    except AssertionError:
        pass


if __name__ == "__main__":
    test_exact_twins_and_test_sets()
    test_validity_in_controls()
    test_default_path_reproduces_stored_results()
    print("all tests passed")
