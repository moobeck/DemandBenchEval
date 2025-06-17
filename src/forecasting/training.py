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
                            "date_features": self._forecast_config.date_features,
                        }
                    ),
                    "fit_config": lambda trial: {
                        "static_features": self._forecast_columns.static, 
                        "max_horizon": self._forecast_config.horizon,
                    },
                },
            ),
            Framework.NEURAL: (NeuralForecastEngine, {}),
            Framework.FM: (FoundationModelEngine, {}),
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
                if fw.value == models_fw.value:  # Compare enum values instead of objects
                    matching_fw = models_fw
                    break
            
            if matching_fw is None or not models_dict[matching_fw]:
                fw_instances[fw] = None
                continue
                
            models = list(models_dict[matching_fw].values())
            if not models:
                fw_instances[fw] = None
                continue

            params = {
                "models": models,
                "freq": Frequency.get_alias(self._forecast_config.freq, 'nixtla'),
                **extra,  # framework-specific kwargs
            }
            fw_instances[fw] = cls(**params)
        return fw_instances

    def cross_validate(
        self,
        df: pd.DataFrame,
        **kwargs,
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
            cv_input = self._prepare_cv_inputs(framework, df, **kwargs)
            df_cv = self._run_framework_cv(engine, **cv_input)
            results.append(df_cv)

        return self._combine_results(results)

    def _prepare_cv_inputs(
        self,
        framework: Framework,
        df: pd.DataFrame,
        **user_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Build the arguments needed for a framework's cross_validation call.
        """
        cols = list(self._forecast_columns.ts_base_cols)
        kwargs: Dict[str, Any] = user_kwargs.copy()

        if framework == Framework.NEURAL:
            # Neural wants static_df separate
            kwargs["static_df"] = self._build_static_df(df)

        elif framework == Framework.FM:
            # Foundation Models need all columns including static and exogenous
            cols += self._forecast_columns.static
            if hasattr(self._forecast_columns, 'exogenous'):
                cols += [col for col in self._forecast_columns.exogenous if col in df.columns]
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
        Concatenate and dedupe columns.
        """
        combined = pd.concat(dfs, axis=1).reset_index()
        # Drop any duplicated forecast columns, keep first
        return combined.loc[:, ~combined.columns.duplicated()].copy()
