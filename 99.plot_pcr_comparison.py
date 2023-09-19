# %% imports and definition
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DPATH_GCAMP = "./intermediate/gcamp_linear_models.dat"
DPATH_GFP = "./intermediate/gfp_linear_models.dat"
FIG_PATH = "./figs/pcr_comparison"
os.makedirs(FIG_PATH, exist_ok=True)

# %% load and construct data
with open(DPATH_GCAMP, "rb") as pklf:
    data_gcamp = pkl.load(pklf)
with open(DPATH_GFP, "rb") as pklf:
    data_gfp = pkl.load(pklf)

result_df = []
for grp, ds in {"GCAMP": data_gcamp, "GFP": data_gfp}.items():
    for dat_key, dat in ds.items():
        for behav_var, behav_dat in dat.items():
            for mod, mod_dat in behav_dat.items():
                if mod == True:
                    mod = "BSN"
                elif mod == False:
                    mod = "Population"
                res = pd.Series(
                    {
                        "group": grp,
                        "dat_key": dat_key,
                        "behav_var": behav_var,
                        "model": mod,
                        "R2ms_train": mod_dat["R2ms_train"],
                        "R2ms_test": mod_dat["R2ms_test"],
                    }
                )
                result_df.append(res)
result_df = pd.concat(result_df, axis="columns", ignore_index=True).T.astype(
    {"R2ms_train": float, "R2ms_test": float}
)

# %% plot results
for yvar in ["R2ms_train", "R2ms_test"]:
    g = sns.FacetGrid(result_df, row="behav_var", col="group", margin_titles=True)
    g.map_dataframe(sns.boxplot, x="model", y=yvar, width=0.4, color="white")
    g.map_dataframe(sns.lineplot, x="model", y=yvar, hue="dat_key")
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(FIG_PATH, "{}.svg".format(yvar)), bbox_inches="tight")
    g.fig.savefig(
        os.path.join(FIG_PATH, "{}.png".format(yvar)), dpi=500, bbox_inches="tight"
    )
