import pytest

from cell_eval2.config import DEParams, DiscriminationParams, EvalConfig, FilterParams


def test_defaults_are_corrected():
    cfg = EvalConfig()
    assert cfg.metrics == "full"
    assert cfg.pert_col == "target"
    assert cfg.control == "non-targeting"
    assert cfg.control_source == "real"          # corrected default
    assert cfg.input_type == "counts"
    assert cfg.max_counts_per_cell == 1_000_000.0
    assert cfg.filter.filter_gene_min_cpm_cell == 5.0
    assert cfg.num_threads == -1
    assert cfg.outdir is None
    assert cfg.discrimination.distance == "cosine"
    assert cfg.discrimination.rank_denominator == "n-1"
    assert cfg.discrimination.exclude_target_gene is True
    assert cfg.discrimination.embed_key is None
    assert cfg == EvalConfig.corrected()


def test_legacy_preset():
    cfg = EvalConfig.legacy()
    assert cfg.control_source == "pred"
    assert cfg.discrimination.distance == "l1"
    assert cfg.discrimination.rank_denominator == "n"
    assert cfg.discrimination.exclude_target_gene is True
    assert cfg != EvalConfig.corrected()


def test_from_dict_overrides_and_nested():
    cfg = EvalConfig.from_dict({"metrics": "vcc", "input_type": "counts",
                                "filter": {"filter_gene_min_cpm_cell": 2.5},
                                "discrimination": {"distance": "l2",
                                                   "rank_denominator": "n"}})
    assert cfg.metrics == "vcc"
    assert cfg.input_type == "counts"
    assert cfg.filter.filter_gene_min_cpm_cell == 2.5
    assert cfg.discrimination.distance == "l2"
    assert cfg.discrimination.rank_denominator == "n"
    assert isinstance(cfg.discrimination, DiscriminationParams)


def test_yaml_round_trip(tmp_path):
    cfg = EvalConfig(metrics=["mae"], control_source="real",
                     filter=FilterParams(filter_gene_min_cpm_cell=1.0),
                     discrimination=DiscriminationParams(distance="l1",
                                                         rank_denominator="n"),
                     outdir=str(tmp_path))
    path = tmp_path / "run_params.yaml"
    cfg.to_yaml(str(path))
    loaded = EvalConfig.from_yaml(str(path))
    assert loaded == cfg


def test_packaged_presets_round_trip():
    assert EvalConfig.from_preset("legacy") == EvalConfig.legacy()
    assert EvalConfig.from_preset("corrected") == EvalConfig.corrected()


def test_packaged_presets_include_new_v1_fields():
    assert EvalConfig.from_preset("v1") == EvalConfig.v1()   # YAML must carry new fields
    assert EvalConfig.from_preset("v2") == EvalConfig.v2()
    v1 = EvalConfig.from_preset("v1")
    assert v1.target_sum is None and v1.de.clip_value == 20.0 and v1.de.fdr_scope == "global"


def test_deparams_defaults_are_v2():
    de = DEParams()
    assert de.p_adj_threshold == 0.05
    assert de.sort_by == "abs_log2_fold_change"
    assert de.method == "wilcoxon"
    assert de.nan_lfc_policy == "mask"


def test_deparams_backend_meancalc_epsilon_defaults():
    from cell_eval2.config import DEParams
    de = DEParams()
    assert de.backend == "auto"
    assert de.mean_calc == "arithmetic"
    assert de.epsilon == 1e-9


def test_deparams_validators_reject_bad_values():
    from cell_eval2.config import DEParams
    with pytest.raises(ValueError):
        DEParams(backend="cpu")
    with pytest.raises(ValueError):
        DEParams(mean_calc="median")
    with pytest.raises(ValueError):
        DEParams(epsilon=-1.0)


def test_config_numeric_validators_reject_nonfinite_and_out_of_range():
    # F1.1/F1.2: numeric config validators must reject NaN/±inf (bare sign comparisons let them
    # through -> silent all-NaN LFC / a disabled scale-limit gate) and out-of-range values. Valid
    # values (including the preset defaults and the negative keep-all filter) must still construct.
    import math

    from cell_eval2.config import DEParams, EvalConfig, FilterParams

    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            DEParams(epsilon=bad)                       # F1.1: epsilon finite >= 0
        with pytest.raises(ValueError):
            DEParams(clip_value=bad)                    # F1.1: clip_value finite > 1 (or None)
        with pytest.raises(ValueError):
            EvalConfig(target_sum=bad)                  # F1.1: target_sum finite > 0 (or None)
        with pytest.raises(ValueError):
            EvalConfig(max_counts_per_cell=bad)         # F1.2: max_counts_per_cell finite > 0
        with pytest.raises(ValueError):
            FilterParams(filter_gene_min_cpm_cell=bad)  # F1.2: finite (or None)
    for bad in (math.nan, math.inf, -0.1, 1.5):
        with pytest.raises(ValueError):
            DEParams(p_adj_threshold=bad)               # F1.2: p_adj_threshold finite in [0, 1]
    with pytest.raises(ValueError):
        EvalConfig(max_counts_per_cell=0.0)             # must be > 0
    for bad in (0, -2, 1.5, math.nan, True):            # F1.2: -1 (all) or a positive int, strictly
        with pytest.raises(ValueError):
            EvalConfig(num_threads=bad)                 # reject 0, <-1, non-int (float/NaN), and bool
    # Valid values still construct:
    DEParams(epsilon=0.0, clip_value=20.0, p_adj_threshold=0.05)
    EvalConfig(target_sum=None, max_counts_per_cell=1e6, num_threads=-1)
    FilterParams(filter_gene_min_cpm_cell=None)
    FilterParams(filter_gene_min_cpm_cell=-1.0)         # negative = keep-all; only NaN/inf rejected


def test_preset_nan_policy_differs():
    assert EvalConfig.legacy().de.nan_lfc_policy == "keep"
    assert EvalConfig.corrected().de.nan_lfc_policy == "mask"
    assert EvalConfig().de.nan_lfc_policy == "mask"  # native default == corrected


def test_deparams_yaml_roundtrip(tmp_path):
    cfg = EvalConfig.legacy()
    p = tmp_path / "c.yaml"
    cfg.to_yaml(str(p))
    back = EvalConfig.from_yaml(str(p))
    assert back.de == cfg.de


def test_deparams_from_dict_coerces_nested():
    cfg = EvalConfig.from_dict({"de": {"p_adj_threshold": 0.1, "nan_lfc_policy": "keep"}})
    assert isinstance(cfg.de, DEParams)
    assert cfg.de.p_adj_threshold == 0.1 and cfg.de.nan_lfc_policy == "keep"


def test_version_default_is_v2():
    assert EvalConfig().version == "v2"
    assert EvalConfig() == EvalConfig.v2()


def test_v2_is_counts_v1_is_lognorm():
    assert EvalConfig().input_type == "counts"          # native default == v2
    assert EvalConfig.v2().input_type == "counts"
    assert EvalConfig.v1().input_type == "lognorm"


def test_target_sum_version_scoped():
    assert EvalConfig().target_sum == 1e6                 # native == v2
    assert EvalConfig.v2().target_sum == 1e6
    assert EvalConfig.v1().target_sum is None             # median, matches cell-eval
    assert EvalConfig() == EvalConfig.v2()                # invariant preserved


def test_clip_value_version_scoped():
    assert EvalConfig().de.clip_value is None         # v2 default
    assert EvalConfig.v1().de.clip_value == 20.0
    assert EvalConfig() == EvalConfig.v2()


def test_fdr_scope_version_scoped():
    assert EvalConfig().de.fdr_scope == "per_pert"     # v2 default
    assert EvalConfig.v1().de.fdr_scope == "global"
    assert EvalConfig() == EvalConfig.v2()


def test_max_counts_per_cell_version_scoped():
    # v1 relaxes the scale-limit gate (cell-eval 0.6.6 has no such gate); v2/default keep 1e6.
    assert EvalConfig().max_counts_per_cell == 1_000_000.0        # v2 default
    assert EvalConfig.v2().max_counts_per_cell == 1_000_000.0
    assert EvalConfig.v1().max_counts_per_cell == 1_000_000_000.0
    assert EvalConfig() == EvalConfig.v2()                        # invariant preserved


def test_allow_discrete_default_false():
    assert EvalConfig().allow_discrete is False
    assert EvalConfig() == EvalConfig.v2()


def test_autodetect_input_type_default_false():
    # orthogonal opt-in: default off, not part of _VERSION_CONVENTIONS, invariant preserved.
    assert EvalConfig().autodetect_input_type is False
    assert EvalConfig.v1().autodetect_input_type is False
    assert EvalConfig.v2().autodetect_input_type is False
    assert EvalConfig() == EvalConfig.v2()


def test_autodetect_input_type_roundtrips_yaml(tmp_path):
    cfg = EvalConfig(autodetect_input_type=True)
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(str(p))
    assert EvalConfig.from_yaml(str(p)).autodetect_input_type is True


def test_v2_mean_calc_is_arithmetic_v1_geometric():
    from cell_eval2.config import EvalConfig
    assert EvalConfig.v2().de.mean_calc == "arithmetic"
    assert EvalConfig.v1().de.mean_calc == "geometric"
    assert EvalConfig() == EvalConfig.v2()              # default still equals v2
    assert EvalConfig().de.mean_calc == "arithmetic"


def test_de_conventions_are_version_scoped():
    v1, v2 = EvalConfig.v1(), EvalConfig.v2()
    assert (v1.de.mean_calc, v1.de.epsilon) == ("geometric", 0.0)
    assert (v2.de.mean_calc, v2.de.epsilon) == ("arithmetic", 1e-9)
    assert v1.filter.filter_gene_min_cpm_cell is None
    assert v2.filter.filter_gene_min_cpm_cell == 5.0


def test_eval_config_default_still_equals_v2():
    assert EvalConfig() == EvalConfig.v2()


def test_v1_v2_presets_and_legacy_aliases():
    v1 = EvalConfig.v1()
    assert v1.version == "v1"
    assert v1.control_source == "pred"
    assert v1.discrimination.distance == "l1" and v1.discrimination.rank_denominator == "n"
    assert v1.de.nan_lfc_policy == "keep"
    assert EvalConfig.legacy() == v1                 # legacy is an alias of v1
    assert EvalConfig.corrected() == EvalConfig.v2() # corrected is an alias of v2
    assert EvalConfig.v2().version == "v2"


def test_from_preset_accepts_v1_v2_and_legacy_names():
    assert EvalConfig.from_preset("v1") == EvalConfig.v1()
    assert EvalConfig.from_preset("v2") == EvalConfig.v2()
    assert EvalConfig.from_preset("legacy") == EvalConfig.v1()
    assert EvalConfig.from_preset("corrected") == EvalConfig.v2()


def test_version_round_trips_in_yaml(tmp_path):
    cfg = EvalConfig.v1()
    p = tmp_path / "c.yaml"
    cfg.to_yaml(str(p))
    assert EvalConfig.from_yaml(str(p)) == cfg


def test_invalid_version_raises():
    with pytest.raises(ValueError, match="version"):
        EvalConfig(version="V1")          # wrong case
    with pytest.raises(ValueError, match="version"):
        EvalConfig.from_dict({"version": "v3"})


def test_invalid_control_source_raises():
    with pytest.raises(ValueError, match="control_source"):
        EvalConfig(control_source="bogus")


def test_invalid_input_type_raises():
    with pytest.raises(ValueError, match="input_type"):
        EvalConfig(input_type="raw")


def test_invalid_distance_raises():
    with pytest.raises(ValueError, match="distance"):
        DiscriminationParams(distance="manhattan")


def test_invalid_rank_denominator_raises():
    with pytest.raises(ValueError, match="rank_denominator"):
        DiscriminationParams(rank_denominator="n+1")


def test_invalid_sort_by_raises():
    with pytest.raises(ValueError, match="sort_by"):
        DEParams(sort_by="pvalue")


def test_invalid_de_method_raises():
    with pytest.raises(ValueError, match="method"):
        DEParams(method="ttest")


def test_invalid_nan_lfc_policy_raises():
    with pytest.raises(ValueError, match="nan_lfc_policy"):
        DEParams(nan_lfc_policy="drop")


def test_valid_presets_still_build():
    for c in (EvalConfig(), EvalConfig.v1(), EvalConfig.v2(),
              EvalConfig.from_preset("v1"), EvalConfig.from_preset("v2")):
        assert c.version in ("v1", "v2")


def test_eval_config_cache_fields_default_off():
    cfg = EvalConfig()
    assert cfg.cache_real is None
    assert cfg.cache_pred is None
    assert cfg.cache_strict is False


def test_eval_config_cache_roundtrips_yaml(tmp_path):
    cfg = EvalConfig(cache_real="/c/real", cache_pred="/c/pred", cache_strict=True)
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(str(p))
    back = EvalConfig.from_yaml(str(p))
    assert back.cache_real == "/c/real"
    assert back.cache_pred == "/c/pred"
    assert back.cache_strict is True


def test_eval_config_rejects_equal_cache_dirs():
    import pytest
    with pytest.raises(ValueError, match="cache_real and cache_pred must differ"):
        EvalConfig(cache_real="/same", cache_pred="/same")


def test_eval_config_rejects_equal_cache_dirs_after_normalization():
    import pytest
    with pytest.raises(ValueError, match="must differ"):
        EvalConfig(cache_real="cache", cache_pred="./cache")  # normalized to the same path


def test_auc_pval_floor_version_scoped():
    assert EvalConfig().de.auc_pval_floor == "min_nonzero"        # v2 default
    assert EvalConfig.v2().de.auc_pval_floor == "min_nonzero"
    assert EvalConfig.v1().de.auc_pval_floor == "replace_zero"    # cell-eval exact
    assert EvalConfig().de.auc_pval_floor_value == 1e-10
    assert EvalConfig.v1().de.auc_pval_floor_value == 1e-10
    assert EvalConfig() == EvalConfig.v2()                        # invariant preserved


def test_auc_pval_floor_validators_reject_bad_values():
    with pytest.raises(ValueError, match="auc_pval_floor"):
        DEParams(auc_pval_floor="floor")
    with pytest.raises(ValueError, match="auc_pval_floor_value"):
        DEParams(auc_pval_floor_value=0.0)
    with pytest.raises(ValueError, match="auc_pval_floor_value"):
        DEParams(auc_pval_floor_value=2.0)


def test_auc_pval_floor_roundtrips_yaml(tmp_path):
    cfg = EvalConfig.v1()
    p = tmp_path / "c.yaml"
    cfg.to_yaml(str(p))
    assert EvalConfig.from_yaml(str(p)).de.auc_pval_floor == "replace_zero"


def test_validate_input_defaults_true_all_versions():
    assert EvalConfig().validate_input is True
    assert EvalConfig.v1().validate_input is True
    assert EvalConfig.v2().validate_input is True


def test_validate_input_can_be_disabled():
    cfg = EvalConfig(validate_input=False)
    assert cfg.validate_input is False


def test_from_dict_rejects_non_mapping():
    """Empty/comment-only YAML parses to None (and scalars to non-dicts); from_dict must
    raise a clear ValueError instead of a cryptic 'NoneType is not iterable'."""
    import pytest

    from cell_eval2.config import EvalConfig
    with pytest.raises(ValueError, match="mapping"):
        EvalConfig.from_dict(None)
    with pytest.raises(ValueError, match="mapping"):
        EvalConfig.from_dict([1, 2, 3])


def test_from_yaml_rejects_empty_file(tmp_path):
    """An empty (comment-only) --config YAML errors clearly rather than crashing."""
    import pytest

    from cell_eval2.config import EvalConfig
    empty = tmp_path / "empty.yaml"
    empty.write_text("# just a comment\n")
    with pytest.raises(ValueError, match="mapping"):
        EvalConfig.from_yaml(str(empty))


def test_shipped_preset_yaml_values_match_to_dict():
    # Every key the shipped preset YAML carries must equal the constructed config's
    # value, so any future per-version default drift on a shipped key is caught.
    # (The shipped YAMLs intentionally omit some default-valued keys; this subset
    # check tolerates the omissions while still pinning the present values.)
    import yaml
    import pathlib
    import cell_eval2
    from cell_eval2.config import EvalConfig
    cfg_dir = pathlib.Path(cell_eval2.__file__).parent / "configs"
    for v in ("v1", "v2"):
        shipped = yaml.safe_load((cfg_dir / f"{v}.yaml").read_text())
        td = EvalConfig.for_version(v).to_dict()
        for k, val in shipped.items():
            assert td[k] == val, f"{v}.yaml[{k}]={val!r} != to_dict()[{k}]={td[k]!r}"


def test_device_and_pert_chunk_defaults():
    cfg = EvalConfig()
    assert cfg.device == "auto"
    assert cfg.pert_chunk == 512
    assert EvalConfig() == EvalConfig.v2()  # new knobs preserve the default==v2 invariant


def test_device_pert_chunk_roundtrip_yaml(tmp_path):
    cfg = EvalConfig(device="cuda", pert_chunk=128)
    p = tmp_path / "c.yaml"
    cfg.to_yaml(str(p))
    back = EvalConfig.from_yaml(str(p))
    assert back.device == "cuda" and back.pert_chunk == 128
    assert back == cfg


def test_device_pert_chunk_from_dict_and_to_dict():
    cfg = EvalConfig.from_dict({"device": "cpu", "pert_chunk": 64})
    assert cfg.device == "cpu" and cfg.pert_chunk == 64
    td = cfg.to_dict()
    assert td["device"] == "cpu" and td["pert_chunk"] == 64


def test_invalid_device_and_pert_chunk_raise():
    with pytest.raises(ValueError, match="device"):
        EvalConfig(device="gpu")
    with pytest.raises(ValueError, match="pert_chunk"):
        EvalConfig(pert_chunk=0)
    with pytest.raises(ValueError, match="pert_chunk"):
        EvalConfig(pert_chunk=-5)


def test_deparams_min_abs_log2fc_default_zero():
    assert DEParams().min_abs_log2fc == 0.0


def test_deparams_min_abs_log2fc_negative_raises():
    with pytest.raises(ValueError, match="min_abs_log2fc"):
        DEParams(min_abs_log2fc=-0.5)


def test_deparams_min_abs_log2fc_nonfinite_raises():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="min_abs_log2fc"):
            DEParams(min_abs_log2fc=bad)


def test_deparams_min_abs_log2fc_zero_ok():
    assert DEParams(min_abs_log2fc=0.0).min_abs_log2fc == 0.0  # boundary, no raise


def test_min_abs_log2fc_yaml_roundtrip():
    cfg = EvalConfig.from_dict({"de": {"min_abs_log2fc": 1.0}})
    assert cfg.de.min_abs_log2fc == 1.0


def test_presets_keep_zero_floor_and_identity():
    # opt-in no-op default: no preset sets the floor, and EvalConfig() == v2().
    assert EvalConfig.v2().de.min_abs_log2fc == 0.0
    assert EvalConfig.v1().de.min_abs_log2fc == 0.0
    assert EvalConfig() == EvalConfig.v2()


def test_gather_threads_validation():
    """gather_threads mirrors num_threads: -1 (auto) or a positive int, strictly."""
    from cell_eval2.config import EvalConfig

    assert EvalConfig().gather_threads == -1
    # float("nan"), NOT math.nan: test_config.py imports math only INSIDE another test
    # function (line 100), so a module-level math.nan here would NameError.
    for bad in (0, -2, 1.5, float("nan"), True):
        with pytest.raises(ValueError):
            EvalConfig(gather_threads=bad)
    EvalConfig(gather_threads=-1)
    EvalConfig(gather_threads=8)


def test_bulk_target_sum_default_and_validation():
    from cell_eval2.config import EvalConfig
    # 5e4 as of #268: 1e6 was the only value on the sweep where the metric broke.
    assert EvalConfig().bulk_target_sum == 50_000.0
    for bad in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="bulk_target_sum"):
            EvalConfig(bulk_target_sum=bad)


def test_bulk_target_sum_enters_the_config_hash():
    from cell_eval2.config import EvalConfig
    from cell_eval2.cache import config_hash
    # config_hash takes the DICT (cache.py:415), not the config object.
    assert (config_hash(EvalConfig().to_dict())
            != config_hash(EvalConfig(bulk_target_sum=28_000.0).to_dict()))


def test_new_fields_are_appended_never_inserted():
    """EvalConfig is a plain (not kw_only) dataclass, so field ORDER is public API: inserting
    a field mid-list would shift device/pert_chunk/outdir for positional callers.

    This asserts the TAIL, not just the last name, so appending stays visible as an
    append: a new field pushes the previous one left by exactly one and displaces nothing
    before it. #195 appended target_gene_map after gather_threads (which #149 had itself
    appended). When you append the next one, extend this list deliberately -- after
    checking the new field really is last and that nothing above it moved."""
    import dataclasses

    from cell_eval2.config import EvalConfig

    assert [f.name for f in dataclasses.fields(EvalConfig)][-3:] == [
        "gather_threads",
        "target_gene_map",
        "bulk_target_sum",
    ]


def test_v1_rejects_the_deseq2_backend():
    """v1 reproduces upstream cell-eval, which has no deseq2 backend, so the combination has
    no referent -- and it was not merely meaningless, it BYPASSED the version gate. Measured:
    `resolve_metrics` blocks the v2-native `de_deseq2_*` names, but `_effective_de_spec`
    relabels the surviving `de_wilcoxon_*` selections into that same family afterwards, so
    `metrics="de"` under v1+deseq2 emitted 21 names, every one of them `v1_available=False`.

    Rejected in `__post_init__` so it binds every driver (run, scale, partition_inmem,
    ceiling, baseline) rather than one call site. Re-filtering after the relabel was the
    alternative and is worse: it empties the DE table instead of saying why.

    EVERY construction route is exercised, not just the direct one. The first version of this
    guard read `self.de.backend` in `__post_init__`, which `from_dict` never reached: it
    popped `de`, built the config from what was left, and assigned the params afterwards, so
    validation saw the DEFAULT backend. `from_yaml` and the CLI's `--set de.backend=deseq2`
    both route through `from_dict`, so all three silently bypassed it.
    """
    import pytest

    from cell_eval2.config import DEParams, EvalConfig

    with pytest.raises(ValueError, match="incompatible with de.backend='deseq2'"):
        EvalConfig(metrics="de", version="v1", de=DEParams(backend="deseq2"))
    with pytest.raises(ValueError, match="incompatible with de.backend='deseq2'"):
        EvalConfig(metrics="de", version="v1", de={"backend": "deseq2"})
    with pytest.raises(ValueError, match="incompatible with de.backend='deseq2'"):
        EvalConfig.from_dict({"metrics": "de", "version": "v1", "de": {"backend": "deseq2"}})


def test_nested_params_are_coerced_whatever_route_built_the_config():
    """`run._resolve_config` documents that a caller may build `EvalConfig(de={...})` with a
    raw dict. Cross-field validation runs in `__post_init__`, so the coercion has to happen
    there too -- reading `.backend` off a plain dict is an AttributeError, and it only shows
    up on the branch that reaches it (v1 raised, v2 short-circuited and looked fine)."""
    from cell_eval2.config import DEParams, DiscriminationParams, EvalConfig, FilterParams

    for version in ("v1", "v2"):
        cfg = EvalConfig(metrics="de", version=version, de={"backend": "pdex"},
                         filter={}, discrimination={})
        assert isinstance(cfg.de, DEParams) and cfg.de.backend == "pdex"
        assert isinstance(cfg.filter, FilterParams)
        assert isinstance(cfg.discrimination, DiscriminationParams)
    # an explicit None means "use the default", by whichever route. Before this it built on
    # v2 and AttributeError'd on v1, because only the v1 branch reached `.backend` (Gemini).
    for version in ("v1", "v2"):
        cfg = EvalConfig(metrics="de", version=version, de=None, filter=None,
                         discrimination=None)
        assert isinstance(cfg.de, DEParams) and cfg.de.backend == "auto"
        assert isinstance(cfg.filter, FilterParams)
        assert isinstance(cfg.discrimination, DiscriminationParams)
    # from_dict keeps working for the params it always handled, including an explicit YAML null
    assert EvalConfig.from_dict({"metrics": "de", "de": None}).de.backend == "auto"
    assert EvalConfig.from_dict(EvalConfig.v1().to_dict()).version == "v1"


def test_the_two_legitimate_backend_version_pairs_still_build():
    from cell_eval2.config import DEParams, EvalConfig

    assert EvalConfig(metrics="de", version="v2", de=DEParams(backend="deseq2")).version == "v2"
    assert EvalConfig(metrics="de", version="v1").de.backend == "auto"
    assert EvalConfig.v1().version == "v1"          # the compat layer's own constructor


def test_a_config_mutated_after_construction_is_caught_at_the_driver_boundary():
    """EvalConfig is mutable and callers do assign fields, so construction-time validation
    alone leaves `cfg = EvalConfig.v1(); cfg.de = DEParams(backend="deseq2")` accepted.
    `_resolve_config` re-runs `__post_init__` via an unconditional `replace`, which is the
    one boundary every driver passes through."""
    import pytest

    from cell_eval2.config import DEParams, EvalConfig
    from cell_eval2.run import _resolve_config

    cfg = EvalConfig.v1()
    cfg.de = DEParams(backend="deseq2")
    with pytest.raises(ValueError, match="incompatible with de.backend='deseq2'"):
        _resolve_config(cfg, {})
    # and a healthy config is unchanged by the revalidation
    assert _resolve_config(EvalConfig.v1(), {}).version == "v1"
    assert _resolve_config(EvalConfig(), {"version": "v1"}).version == "v1"
