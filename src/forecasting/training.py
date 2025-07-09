import logging
import pandas as pd
from typing import Any, Dict, List
from src.forecasting.engine import (
    ForecastEngine,
    StatsForecastEngine,
    AutoMLForecastEngine,
    NeuralForecastEngine,
    FoundationModelEngine,
)
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.forecasting import ForecastConfig
from src.configurations.enums import Framework, Frequency
from src.configurations.cross_validation import CrossValDatasetConfig
from mlforecast.target_transforms import LocalMinMaxScaler


class ForecastTrainer:
    """
    Orchestrates model fitting
    for multiple models from multiple frameworks
    """

    def __init__(
        self,
        forecast_config: ForecastConfig,
        forecast_columns: ForecastColumnConfig = None,
    ):

        self._forecast_config = forecast_config
        self._forecast_columns = forecast_columns

        self._factory = {
            Framework.STATS: (StatsForecastEngine, {}),
            Framework.ML: (
                AutoMLForecastEngine,
                {
                    "init_config": lambda trial: (
                        {
                            "lags": self._forecast_config.lags,
                            "target_transforms": [LocalMinMaxScaler()],
                        }
                    ),
                    "fit_config": lambda trial: {
                        "static_features": self._forecast_columns.static,
                        "max_horizon": self._forecast_config.horizon,
                    },
                    "num_samples": forecast_config.model_config[Framework.ML][
                        "num_samples"
                    ] if Framework.ML in forecast_config.model_config else None,
                },
            ),
            Framework.NEURAL: (NeuralForecastEngine, {}),
            Framework.FM: (
                FoundationModelEngine,
                {},
            ),
        }

        self.frameworks = self._build_frameworks()

    def _build_frameworks(self) -> Dict[Framework, ForecastEngine]:
        fw_instances = {}
        models_dict = self._forecast_config.models  # Access models once and cache

        for fw, (cls, extra) in self._factory.items():
            # Check if framework has models using a more robust comparison
            # Find matching framework by value instead of object identity
            matching_fw = None
            for models_fw in models_dict.keys():
                if (
                    fw.value == models_fw.value
                ):  # Compare enum values instead of objects
                    matching_fw = models_fw
                    break

            if matching_fw is None or not models_dict[matching_fw]:
                fw_instances[fw] = None
                continue

            models = list(models_dict.get(fw, {}).values())
            if not models:
                fw_instances[fw] = None
                continue

            params = {
                "models": models,
                "freq": Frequency.get_alias(self._forecast_config.freq, "nixtla"),
                **extra,  # framework-specific kwargs
            }
            fw_instances[fw] = cls(**params)
        return fw_instances

    def cross_validate(
        self,
        df: pd.DataFrame,
        cv_config: CrossValDatasetConfig
    ) -> pd.DataFrame:
        """
        Perform cross-validation for each forecasting framework and
        return a combined DataFrame of predictions.
        """
        logging.info("Starting cross-validation...")
        results = []

        for framework, engine in self.frameworks.items():
            if engine is None:
                continue

            logging.info(f"Cross-validating with {framework.name}...")
            cv_input = self._prepare_cv_inputs(framework, df, cv_config=cv_config)
            df_cv = self._run_framework_cv(engine, **cv_input)
            results.append(df_cv)

        combined_results = self._combine_results(results)
        logging.info("Cross-validation completed.")

        return combined_results

    def _prepare_cv_inputs(
        self,
        framework: Framework,
        df: pd.DataFrame,
        cv_config: CrossValDatasetConfig,
    ) -> Dict[str, Any]:
        """
        Build the arguments needed for a framework's cross_validation call.
        """
        cols = list(self._forecast_columns.ts_base_cols)
        kwargs: Dict[str, Any] = {"cv_config": cv_config}

        if framework == Framework.NEURAL:
            # Neural wants static_df separate
            kwargs["static_df"] = self._build_static_df(df)

        elif framework == Framework.FM:
            # Foundation Models need all columns including static and exogenous
            cols += self._forecast_columns.static
            if self._forecast_columns.exogenous:
                cols += [
                    col for col in self._forecast_columns.exogenous if col in df.columns
                ]
            kwargs["forecast_columns"] = self._forecast_columns
            kwargs["forecast_config"] = self._forecast_config
            kwargs["h"] = self._forecast_config.horizon

        else:
            kwargs["h"] = self._forecast_config.horizon

            if framework == Framework.ML:
                # ML wants static_features inline
                cols += self._forecast_columns.static
                kwargs["forecast_columns"] = self._forecast_columns
                kwargs["forecast_config"] = self._forecast_config

        return {
            "df": df[cols],
            "id_col": self._forecast_columns.sku_index,
            "target_col": self._forecast_columns.target,
            "time_col": self._forecast_columns.date,
            **kwargs,
        }

    def _build_static_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build the static_df for the NeuralForecast framework.
        """
        return (
            df[[self._forecast_columns.sku_index] + self._forecast_columns.static]
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
        df_out = engine.cross_validation(**cv_kwargs)

        return df_out.set_index(
            [self._forecast_columns.sku_index, self._forecast_columns.date],
            drop=True,
        )

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
                logging.warning(f"DataFrame {i} has {df.index.duplicated().sum()} duplicate indices")
                # Remove duplicates, keeping the first occurrence
                dfs[i] = df[~df.index.duplicated(keep='first')]
        
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
                index_cols = df.columns[:2].tolist()  # Assuming 2-level index (sku_index, date)
                base_df = pd.merge(base_df, df, on=index_cols, how='outer')
            
            combined = base_df
        
        # Drop any duplicated forecast columns, keep first
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined
