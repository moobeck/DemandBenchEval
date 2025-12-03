import logging
import pandas as pd
from typing import Any, Dict, List, Optional
from copy import deepcopy
from src.forecasting.engine.abstract import ForecastEngine
from src.forecasting.engine.statistical import StatsForecastEngine
from src.forecasting.engine.neural import NeuralForecastEngine
from src.forecasting.engine.foundation import FoundationModelEngine
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.forecasting.forecasting import ForecastConfig
from src.configurations.utils.enums import Framework, FrequencyType
from src.configurations.evaluation.cross_validation import CrossValidationConfig
from src.configurations.forecasting.models.model import MODEL_REGISTRY


class CrossValidator:
    """
    Class to manage cross-validation across multiple forecasting frameworks.
    """

    def __init__(
        self,
        forecast_config: ForecastConfig,
        forecast_columns: ForecastColumnConfig,
        cross_validation: CrossValidationConfig,
    ):

        self._forecast_config = forecast_config
        self._forecast_columns = forecast_columns
        self._cross_validation = cross_validation

        self._factory = {
            Framework.STATS: (StatsForecastEngine, {}),
            Framework.NEURAL: (
                NeuralForecastEngine,
                {},
            ),
            Framework.FM: (
                FoundationModelEngine,
                {},
            ),
        }

        self.frameworks = self._build_frameworks()

    def _build_frameworks(self) -> Dict[Framework, ForecastEngine]:
        fw_instances = {}
        for fw in self._factory:
            fw_instances[fw] = self._create_engine(fw)
        return fw_instances

    def _create_engine(self, fw: Framework) -> Optional[ForecastEngine]:
        """
        Create a forecasting engine for the given framework, or return None if no models are configured.
        """
        models_dict = self._forecast_config.models

        matching_fw = None
        for models_fw in models_dict.keys():
            if fw.value == models_fw.value:
                matching_fw = models_fw
                break

        if matching_fw is None or not models_dict[matching_fw]:
            return None

        models = list(models_dict.get(fw, {}).values())
        if not models:
            return None

        cls, extra = self._factory[fw]
        params = {
            "models": models,
            "freq": FrequencyType.get_alias(self._forecast_config.freq, "nixtla"),
            **extra,
        }
        return cls(**params)

    def cross_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform cross-validation for each forecasting framework and
        return a combined DataFrame of predictions.
        """
        logging.info("Starting cross-validation...")

        results = []

        for framework, engine in self.frameworks.items():
            logging.info(f"Framework: {framework}, Engine: {engine}")

            if engine is None:
                logging.info(f"No models defined for framework {framework}. Skipping.")
                continue

            logging.info(f"Cross-validating with {framework.name}...")
            cv_input = self._prepare_cv_inputs(framework, engine, df)
            df_cv = self._run_framework_cv(engine, **cv_input)
            results.append(df_cv)

        combined_results = self._combine_results(results)
        logging.info("Cross-validation completed.")

        return combined_results

    def _prepare_cv_inputs(
        self,
        framework: Framework,
        engine: ForecastEngine,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Build the arguments needed for a framework's cross_validation call.
        """

        exo_cols = (
            self._forecast_columns.future_exogenous
            + self._forecast_columns.past_exogenous
        )
        static_cols = list(self._forecast_columns.static)
        base_cols = cols = list(self._forecast_columns.ts_base_cols)
        cols = base_cols

        if framework == Framework.NEURAL:
            cols += exo_cols + static_cols
        elif framework == Framework.FM:
            cols += (exo_cols + static_cols)

        df = df[cols]

        # For neural models, pad short series with zeros and shrink n_windows to avoid window failures.
        cv_cfg = self._cross_validation
        if framework == Framework.NEURAL:
            id_col = self._forecast_columns.time_series_index
            date_col = self._forecast_columns.date
            min_len = df[id_col].value_counts().min()

            horizon = self._forecast_config.horizon
            # Pad to the maximum input size implied by the search space/default configs
            # so every sampled window has enough history.
            max_input_size = self._max_neural_input_size()
            min_required_len = max_input_size + horizon

            df = self._pad_short_series(
                df,
                id_col=id_col,
                date_col=date_col,
                min_required_len=min_required_len,
                freq_alias=FrequencyType.get_alias(
                    self._forecast_config.freq, "pandas"
                ),
            )
            min_len = df[id_col].value_counts().min()
            min_input_size = max_input_size

            feasible_windows = max(
                1,
                min(
                    cv_cfg.n_windows,
                    int((min_len - min_input_size - horizon) / cv_cfg.step_size) + 1,
                ),
            )
            if feasible_windows < cv_cfg.n_windows:
                logging.warning(
                    "Reducing n_windows from %d to %d to fit shortest series (len=%d).",
                    cv_cfg.n_windows,
                    feasible_windows,
                    min_len,
                )
            cv_cfg = deepcopy(cv_cfg)
            cv_cfg.n_windows = feasible_windows

        possible_inputs = {
            "df": df,
            "cv_config": cv_cfg,
            "forecast_columns": self._forecast_columns,
            "forecast_config": self._forecast_config,
            "h": self._forecast_config.horizon,
            "static_df": self._build_static_df(df),

        }

        return {
            k: v
            for k, v in possible_inputs.items()
            if k in engine.cv_inputs()
        }

    def _max_neural_input_size(self) -> int:
        """
        Compute a conservative upper bound for neural input_size so padding/windows are feasible.
        Mirrors the search space used in ForecastConfig for optuna backends and falls back to
        model defaults when present.
        """
        horizon = self._forecast_config.horizon
        # Matches the safe search space constructed in ForecastConfig for optuna.
        max_size = max(max(4, horizon // 2), max(6, horizon), min(14, 2 * horizon))

        for model_name in self._forecast_config.names:
            spec = MODEL_REGISTRY.get(model_name)
            if spec is None or spec.framework != Framework.NEURAL:
                continue
            default_cfg = spec.model.get_default_config(
                h=horizon, backend="not_specified"
            )
            candidate = default_cfg.get("input_size")
            if isinstance(candidate, int):
                max_size = max(max_size, candidate)

        return max_size

    @staticmethod
    def _pad_short_series(
        df: pd.DataFrame,
        id_col: str,
        date_col: str,
        min_required_len: int,
        freq_alias: str,
    ) -> pd.DataFrame:
        """
        Prepend zero rows for series shorter than the required length.
        """
        lengths = df[id_col].value_counts()
        to_pad = lengths[lengths < min_required_len]
        if to_pad.empty:
            return df

        pad_frames = []
        zero_template = {col: 0 for col in df.columns}
        zero_template[id_col] = None
        zero_template[date_col] = None

        for ts_id, length in to_pad.items():
            subset = df[df[id_col] == ts_id]
            first_date = subset[date_col].min()
            needed = min_required_len - length
            pad_dates = pd.date_range(
                end=first_date, periods=needed + 1, freq=freq_alias
            )[:-1]
            if pad_dates.empty:
                continue
            pad_df = pd.DataFrame(
                {
                    **zero_template,
                    id_col: ts_id,
                    date_col: pad_dates,
                }
            )
            pad_frames.append(pad_df)

        if pad_frames:
            df = pd.concat([df] + pad_frames, ignore_index=True)
            df = df.sort_values([id_col, date_col]).reset_index(drop=True)

        return df

    def _build_static_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build the static_df for the NeuralForecast framework.
        """
        return (
            df[
                [self._forecast_columns.time_series_index]
                + self._forecast_columns.static
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _run_framework_cv(
        self,
        engine: ForecastEngine,
        **cv_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Call the engine's cross_validation and set the proper index.
        """
        try:
            df_out = engine.cross_validation(**cv_kwargs)
        except RuntimeError as exc:
            # Fallback to CPU if a GPU failure occurs during neural training.
            if isinstance(engine, NeuralForecastEngine) and "CUDA" in str(exc):
                logging.error(
                    "CUDA error during neural cross-validation (%s). Retrying on CPU.",
                    exc,
                )
                # Force neural models to use CPU and rebuild the engine.
                self._forecast_config.model_config[Framework.NEURAL]["gpus"] = 0
                new_engine = self._create_engine(Framework.NEURAL)
                if new_engine is None:
                    raise
                self.frameworks[Framework.NEURAL] = new_engine
                df_out = new_engine.cross_validation(**cv_kwargs)
            else:
                raise

        df_out = df_out.set_index(
            [self._forecast_columns.time_series_index, self._forecast_columns.date],
            drop=True,
        )

        # Guard against degenerate runs (e.g., all-NaN predictions from failed trials)
        if df_out.empty or df_out.isna().all().all():
            raise ValueError(
                "Cross-validation returned empty or all-NaN predictions. "
                "Check data quality and CV/hyperparameter settings (e.g., reduce input_size/step_size)."
            )

        # Ensure prediction columns contain actual forecasts (not just targets/meta)
        meta_cols = set(self._forecast_columns.ts_base_cols)
        meta_cols.add(getattr(self._forecast_columns, "cutoff", "cutoff"))
        pred_cols = [c for c in df_out.columns if c not in meta_cols]

        if not pred_cols:
            raise ValueError(
                "Cross-validation produced no prediction columns. "
                "Verify models emit forecasts (e.g., increase quantiles, horizons) and CV settings."
            )

        # Drop models whose prediction columns are all-NaN; keep models with any signal.
        model_cols: Dict[str, List[str]] = {}
        for col in pred_cols:
            base = col.split("-")[0]
            model_cols.setdefault(base, []).append(col)

        models_to_drop = []
        for model, cols in model_cols.items():
            if not df_out[cols].notna().any().any():
                models_to_drop.extend(cols)
                logging.warning(
                    "Dropping model %s (all prediction columns are NaN): %s",
                    model,
                    cols,
                )

        if models_to_drop:
            df_out = df_out.drop(columns=models_to_drop)

        remaining_pred_cols = [c for c in df_out.columns if c not in meta_cols]
        if not remaining_pred_cols:
            null_ratios = (
                df_out.reindex(columns=pred_cols)
                .isna()
                .mean()
                .fillna(1.0)
                .to_dict()
            )
            raise ValueError(
                "Cross-validation produced only NaN prediction columns. "
                f"Checked columns: {pred_cols}. Non-null ratios: {null_ratios}. "
                "Adjust search space or CV settings (smaller input_size/step_size, validate data)."
            )

        return df_out

    @staticmethod
    def _combine_results(
        dfs: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Concatenate and dedupe columns with robust duplicate handling.
        """
        if not dfs:
            return pd.DataFrame()

        if len(dfs) == 1:
            return dfs[0].reset_index()

        # Check for duplicate indices in each DataFrame and clean them
        for i, df in enumerate(dfs):
            if df.index.duplicated().any():
                logging.warning(
                    f"DataFrame {i} has {df.index.duplicated().sum()} duplicate indices"
                )
                # Remove duplicates, keeping the first occurrence
                dfs[i] = df[~df.index.duplicated(keep="first")]

        try:
            combined = pd.concat(dfs, axis=1).reset_index()
        except pd.errors.InvalidIndexError as e:
            logging.error(f"Failed to concatenate DataFrames with axis=1: {e}")
            logging.info("Using outer join to align indices...")

            # Reset indices to columns for all DataFrames
            dfs_reset = [df.reset_index() for df in dfs]

            # Merge DataFrames on the index columns instead of concatenating
            base_df = dfs_reset[0]
            for df in dfs_reset[1:]:
                # Get the index column names (should be the first columns after reset_index)
                index_cols = df.columns[
                    :2
                ].tolist()  # Assuming 2-level index (time_series_index, date)
                base_df = pd.merge(base_df, df, on=index_cols, how="outer")

            combined = base_df

        # Drop any duplicated forecast columns, keep first
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined
