"""Validate experiment settings before a run can download or overwrite artifacts."""

import math


def validate_session_source(session: dict, options: dict | None = None) -> None:
    source = session.get("source", "fastf1")
    if source not in ("fastf1", "openf1"):
        raise ValueError("session.source must be 'fastf1' or 'openf1'.")
    if source == "fastf1":
        return
    if session["year"] < 2023:
        raise ValueError("OpenF1 historical data starts in 2023; this year is unsupported.")
    key = session.get("session_key")
    if isinstance(key, bool) or not isinstance(key, int) or key <= 0:
        raise ValueError("OpenF1 requires a positive integer session.session_key.")
    if not isinstance(options, dict):
        raise ValueError("OpenF1 requires an openf1 configuration mapping.")
    if options.get("base_url") != "https://api.openf1.org/v1":
        raise ValueError("openf1.base_url must be https://api.openf1.org/v1.")
    for name in ("timeout_sec", "max_attempts", "chunk_seconds", "request_interval_sec"):
        item = options.get(name)
        if (isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(item) or item < 0
                or (name != "request_interval_sec" and item == 0)
                or (name in ("max_attempts", "chunk_seconds") and not isinstance(item, int))):
            raise ValueError(f"Invalid or missing openf1.{name}: {item!r}")
    if not isinstance(options.get("fetch_location"), bool):
        raise ValueError("openf1.fetch_location must be a boolean.")


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping.")

    def value(path):
        current = config
        for component in path.split("."):
            if not isinstance(current, dict):
                raise ValueError(f"Configuration section for {path} must be a mapping.")
            if component not in current:
                raise ValueError(f"Missing required configuration key: {path}")
            current = current[component]
        return current

    def number(path, minimum=0, integer=False, strict=False, maximum=None):
        item = value(path)
        if (isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(item) or (integer and not isinstance(item, int))
                or (item <= minimum if strict else item < minimum)
                or (maximum is not None and item > maximum)):
            raise ValueError(f"Invalid numeric configuration value for {path}: {item!r}")
        return item

    for path in ("session.gp", "session.session_type", "session.cache_dir", "session.processed_dir",
                 "train.checkpoint_path"):
        if not isinstance(value(path), str) or not value(path).strip():
            raise ValueError(f"{path} must be a non-empty string.")
    if config.get("train", {}).get("loss", "mae") not in ("mae", "mse"):
        raise ValueError("train.loss must be 'mae' or 'mse'.")
    number("session.year", minimum=1, integer=True)
    validate_session_source(config["session"], config.get("openf1"))
    for path in ("track.mini_sectors", "graph.num_nodes", "data.window_size", "train.batch_size",
                 "train.max_epochs", "train.early_stopping_patience", "model.in_dim", "model.hidden_dim",
                 "model.gat_heads", "model.edge_dim", "model.conv1d_kernel",
                 "target.min_clean_samples", "data.min_valid_targets_per_sample",
                 "baselines.xgboost.n_estimators", "baselines.xgboost.max_depth",
                 "baselines.lstm.hidden_dim", "baselines.lstm.num_layers",
                 "baselines.lstm.epochs", "baselines.lstm.batch_size"):
        number(path, minimum=1, integer=True)
    for path in ("data.horizon", "data.random_seed", "train.num_workers"):
        number(path, integer=True)
    number("data.random_seed", integer=True, maximum=2**32 - 1)
    for path in ("track.lap_length_m", "preprocess.sample_hz", "graph.edge_threshold_sec",
                 "graph.rbf_sigma", "train.lr", "baselines.xgboost.learning_rate",
                 "baselines.lstm.lr", "physics_tests.drs_max_gap_sec"):
        number(path, strict=True)
    for path in ("preprocess.max_interpolate_sec", "target.clean_air_gap_sec",
                 "data.timestamp_tolerance_sec", "data.continuity_tolerance_sec", "train.weight_decay",
                 "physics_tests.clean_air_max_mae", "physics_tests.drs_train_min_lap_loss",
                 "physics_tests.drs_train_max_lap_loss"):
        number(path)
    for path in ("preprocess.flag_safety_car", "preprocess.flag_pit_laps"):
        if not isinstance(value(path), bool):
            raise ValueError(f"{path} must be a boolean.")
    for path in ("model.dropout", "baselines.lstm.dropout"):
        if number(path, maximum=1) == 1:
            raise ValueError(f"{path} must be less than 1.")
    number("baselines.xgboost.subsample", strict=True, maximum=1)

    ratios = [number(f"data.{name}_ratio", strict=True, maximum=1) for name in ("train", "val", "test")]
    if not math.isclose(sum(ratios), 1.0, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("data.train_ratio, data.val_ratio, and data.test_ratio must sum to 1.")
    if value("data.split_method") not in ("temporal", "race"):
        raise ValueError("data.split_method must be 'temporal' or 'race'.")
    if value("data.min_valid_targets_per_sample") > value("graph.num_nodes"):
        raise ValueError("data.min_valid_targets_per_sample cannot exceed graph.num_nodes.")
    if value("model.hidden_dim") % value("model.gat_heads"):
        raise ValueError("model.hidden_dim must be divisible by model.gat_heads.")
    temporal_mode = config["model"].get("temporal_mode", "conv")
    if temporal_mode not in ("conv", "gated_dilated"):
        raise ValueError("model.temporal_mode must be 'conv' or 'gated_dilated'.")
    dilations = config["model"].get("temporal_dilations", [1, 2, 4])
    if (not isinstance(dilations, list) or not dilations
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in dilations)):
        raise ValueError("model.temporal_dilations must be a non-empty list of positive integers.")
    if temporal_mode == "gated_dilated":
        receptive_field = 1 + 2 * sum(dilations)
        if receptive_field < value("data.window_size"):
            raise ValueError("model.temporal_dilations must provide a receptive field covering data.window_size.")
    graph_cfg = value("graph")
    node_schema = graph_cfg.get("node_feature_schema", "legacy_v1")
    if node_schema not in ("legacy_v1", "circuit_context_v1"):
        raise ValueError("graph.node_feature_schema must be 'legacy_v1' or 'circuit_context_v1'.")
    edge_schema = graph_cfg.get("edge_feature_schema", "scalar_rbf")
    if edge_schema not in ("scalar_rbf", "physical_v1"):
        raise ValueError("graph.edge_feature_schema must be 'scalar_rbf' or 'physical_v1'.")
    neighbor_mode = graph_cfg.get("neighbor_mode", "race_progress")
    if neighbor_mode not in ("race_progress", "physical_circuit"):
        raise ValueError("graph.neighbor_mode must be 'race_progress' or 'physical_circuit'.")
    if edge_schema == "physical_v1" and neighbor_mode != "physical_circuit":
        raise ValueError("graph.edge_feature_schema='physical_v1' requires neighbor_mode='physical_circuit'.")
    expected_edge_dim = 1 if edge_schema == "scalar_rbf" else 4
    if value("model.edge_dim") != expected_edge_dim:
        raise ValueError(f"model.edge_dim must be {expected_edge_dim} for graph.edge_feature_schema={edge_schema!r}.")
    categories = value("graph.compound_categories")
    if (not isinstance(categories, list) or not categories
            or any(not isinstance(item, str) or not item.strip() for item in categories)
            or len({item.upper() for item in categories}) != len(categories)):
        raise ValueError("graph.compound_categories must contain distinct, non-empty strings.")
    expected_node_dim = (4 + len(categories)) + (10 if node_schema == "circuit_context_v1" else 0)
    if value("model.in_dim") != expected_node_dim:
        raise ValueError(
            f"model.in_dim must be {expected_node_dim} for graph.node_feature_schema={node_schema!r}."
        )
    clip = value("target.clip_delta_sec")
    if (not isinstance(clip, list) or len(clip) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) for item in clip) or clip[0] >= clip[1]):
        raise ValueError("target.clip_delta_sec must contain two increasing finite numbers.")
    if value("physics_tests.drs_train_min_lap_loss") > value("physics_tests.drs_train_max_lap_loss"):
        raise ValueError("physics_tests.drs_train_min_lap_loss exceeds physics_tests.drs_train_max_lap_loss.")
    tuning = config.get("tuning", {})
    if not isinstance(tuning, dict) or not isinstance(tuning.get("enabled", False), bool):
        raise ValueError("tuning must be a mapping with a boolean tuning.enabled flag.")
