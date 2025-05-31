# %%
import pandas as pd

df = pd.read_feather(
    "/Users/moritzbeckmail.de/Documents/Phd/Research/NixtlaForecast/data/preprocessed_data.feather"
)

df

# %% [markdown]
# cross_validation:
#   cv_windows: 13
#   step_size: 14
#   refit: false

# %%
n_windows = 6
step_size = 1  # days
offset = pd.Timedelta(days=n_windows * step_size)

df_fit = df[df["date"] < df["date"].max() - offset]

# %%
import pandas as pd

df_fit = pd.read_csv("df_fit.csv", parse_dates=["date"])
df_fit

# %%
import pickle

# Load dict_fit

with open("dict_fit.pkl", "rb") as f:
    dict_fit = pickle.load(f)

dict_fit

# %%
from mlforecast.auto import AutoMLForecast, AutoLightGBM

model = AutoLightGBM()

auto_ml_frcast = AutoMLForecast(
    models=[model],
    init_config=lambda x: {
        "lags": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "date_features": ["dayofweek", "month"],
    },
    fit_config=lambda x: {
        "static_features": ["storeID", "productID", "companyID"],
    },
    num_threads=1,
    freq="D",
)

auto_ml_frcast.fit(**dict_fit)


# %%
df = auto_ml_frcast.models_["AutoLightGBM"].cross_validation(
    df,
    n_windows=4,
    h=14,
    id_col="skuID",
    time_col="date",
    target_col="demand",
    static_features=["storeID", "productID", "companyID"],
)

# %%
df
