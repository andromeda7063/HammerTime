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
    if config.get("train", {}).get("loss", "mae") not in ("mae", "mse", "smooth_l1", "huber"):
        raise ValueError("train.loss must be 'mae', 'mse', or 'smooth_l1'.")
    lr_scheduler = config.get("train", {}).get("lr_scheduler", "none")
    if lr_scheduler not in ("none", "plateau"):
        raise ValueError("train.lr_scheduler must be 'none' or 'plateau'.")
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
    if "window_stride" in config.get("data", {}):
        number("data.window_stride", minimum=1, integer=True)
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
    spatial_encoder = config.get("model", {}).get("spatial_encoder", "gat")
    if spatial_encoder not in ("gat", "edge_temporal"):
        raise ValueError("model.spatial_encoder must be 'gat' or 'edge_temporal'.")
    spatial_layers = config.get("model", {}).get("spatial_layers", 2)
    allowed_spatial_layers = (2, 3) if spatial_encoder == "gat" else (0,)
    if (isinstance(spatial_layers, bool) or not isinstance(spatial_layers, int)
            or spatial_layers not in allowed_spatial_layers):
        raise ValueError(f"model.spatial_layers must be one of {allowed_spatial_layers} for {spatial_encoder}.")
    own_car_branch = config.get("model", {}).get("own_car_branch", False)
    if not isinstance(own_car_branch, bool):
        raise ValueError("model.own_car_branch must be a boolean.")
    if "own_car_hidden_dim" in config.get("model", {}):
        number("model.own_car_hidden_dim", minimum=1, integer=True)
    gat_type = config.get("model", {}).get("gat_type", "gat")
    if gat_type not in ("gat", "gatv2"):
        raise ValueError("model.gat_type must be 'gat' or 'gatv2'.")
    if "typed_edge_values" in config.get("model", {}) and not isinstance(
            config["model"]["typed_edge_values"], bool):
        raise ValueError("model.typed_edge_values must be a boolean.")
    typed_edge_schema = config.get("graph", {}).get("edge_feature_schema", "scalar_rbf")
    if config.get("model", {}).get("typed_edge_values", False) and typed_edge_schema != "physical_v2":
        raise ValueError("model.typed_edge_values requires graph.edge_feature_schema='physical_v2'.")
    if "mask_inactive_nodes" in config.get("model", {}) and not isinstance(
            config["model"]["mask_inactive_nodes"], bool):
        raise ValueError("model.mask_inactive_nodes must be a boolean.")
    if config.get("model", {}).get("typed_edge_values", False) and not config.get("model", {}).get(
            "mask_inactive_nodes", False):
        raise ValueError("model.typed_edge_values requires model.mask_inactive_nodes=true.")
    if spatial_encoder == "edge_temporal" and (
            not config.get("model", {}).get("typed_edge_values", False)
            or typed_edge_schema != "physical_v2"
            or not config.get("model", {}).get("mask_inactive_nodes", False)):
        raise ValueError("edge_temporal requires typed_edge_values, physical_v2 edges, and inactive-node masking.")
    if "predict_future_speed" in config.get("model", {}) and not isinstance(
            config["model"]["predict_future_speed"], bool):
        raise ValueError("model.predict_future_speed must be a boolean.")
    if config.get("model", {}).get("predict_future_speed", False):
        horizon = config.get("model", {}).get("future_speed_horizon_steps")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon != value("data.horizon"):
            raise ValueError("model.future_speed_horizon_steps must equal data.horizon when future-speed prediction is enabled.")
        auxiliary = config.get("auxiliary", {})
        if auxiliary.get("future_speed_enabled") is not True:
            raise ValueError("model.predict_future_speed requires auxiliary.future_speed_enabled=true.")
        number("auxiliary.future_speed_weight", strict=True)
        number("auxiliary.future_speed_scale_kph", strict=True)
    elif config.get("auxiliary", {}).get("future_speed_enabled", False):
        raise ValueError("auxiliary.future_speed_enabled requires model.predict_future_speed=true.")
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
    if node_schema not in ("legacy_v1", "circuit_context_v1", "circuit_context_v2"):
        raise ValueError("graph.node_feature_schema must be 'legacy_v1', 'circuit_context_v1', or 'circuit_context_v2'.")
    edge_schema = graph_cfg.get("edge_feature_schema", "scalar_rbf")
    if edge_schema not in ("scalar_rbf", "physical_v1", "physical_v2"):
        raise ValueError("graph.edge_feature_schema must be 'scalar_rbf', 'physical_v1', or 'physical_v2'.")
    neighbor_mode = graph_cfg.get("neighbor_mode", "race_progress")
    if neighbor_mode not in ("race_progress", "physical_circuit"):
        raise ValueError("graph.neighbor_mode must be 'race_progress' or 'physical_circuit'.")
    if edge_schema in ("physical_v1", "physical_v2") and neighbor_mode != "physical_circuit":
        raise ValueError("graph.edge_feature_schema='physical_v1' or 'physical_v2' requires neighbor_mode='physical_circuit'.")
    expected_edge_dim = 1 if edge_schema == "scalar_rbf" else (4 if edge_schema == "physical_v1" else 5)
    if value("model.edge_dim") != expected_edge_dim:
        raise ValueError(f"model.edge_dim must be {expected_edge_dim} for graph.edge_feature_schema={edge_schema!r}.")
    categories = value("graph.compound_categories")
    if (not isinstance(categories, list) or not categories
            or any(not isinstance(item, str) or not item.strip() for item in categories)
            or len({item.upper() for item in categories}) != len(categories)):
        raise ValueError("graph.compound_categories must contain distinct, non-empty strings.")
    context_extra = 10 if node_schema == "circuit_context_v1" else (11 if node_schema == "circuit_context_v2" else 0)
    expected_node_dim = (4 + len(categories)) + context_extra
    if value("model.in_dim") != expected_node_dim:
        raise ValueError(
            f"model.in_dim must be {expected_node_dim} for graph.node_feature_schema={node_schema!r}."
        )
    if "drop_edge_rate" in config.get("model", {}):
        rate = config["model"]["drop_edge_rate"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not (0.0 <= rate < 1.0):
            raise ValueError("model.drop_edge_rate must be a number in [0, 1).")
    if "use_graph_norm" in config.get("model", {}):
        if not isinstance(config["model"]["use_graph_norm"], bool):
            raise ValueError("model.use_graph_norm must be a boolean.")
    if "residual" in config.get("model", {}):
        if not isinstance(config["model"]["residual"], bool):
            raise ValueError("model.residual must be a boolean.")
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
