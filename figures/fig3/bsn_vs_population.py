import os
import pickle

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from utility import user_tracker

behavior = "velocity"
with open(
    "%s/intermediate/gcamp_linear_models.dat" % user_tracker.codePath(), "rb"
) as handle:
    data = pickle.load(handle)

keys = list(data.keys())
keys.sort()

figtypes = ["bsn_deriv", "slm_with_derivs"]

outputFolder = os.path.join(user_tracker.codePath(), "figures/output")
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

outfilename = outputFolder + "/BSN_vs_population_" + behavior + ".pdf"
pdf = matplotlib.backends.backend_pdf.PdfPages(outfilename)


def calc_pdf(x, low_lim, high_lim, nbins):
    counts, bin_edges = np.histogram(x, np.linspace(low_lim, high_lim, nbins))
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    density = np.true_divide(counts, np.sum(counts))
    return density, bin_centers, bin_edges


def frac_range_covered(true, pred, percentile=True):
    if percentile:
        P = 99
        gap_top = np.max(np.append(np.percentile(true, P) - np.percentile(pred, P), 0))
        gap_bottom = np.max(
            np.append(np.percentile(pred, 100 - P) - np.percentile(true, 100 - P), 0)
        )
    else:
        gap_top = np.max(np.append(np.max(true) - np.max(pred), 0))
        gap_bottom = np.max(np.append(np.min(pred) - np.min(true), 0))
    assert gap_bottom >= 0
    assert gap_top >= 0
    range_true = np.ptp(true)
    frac = np.true_divide(range_true - gap_top - gap_bottom, range_true)
    return frac


test_color = "#2ca02c"
pred_color = "#1f77b4"
measured_color = "#C1804A" if behavior == "curvature" else "k"

alpha = 0.5
alpha_test = 0.3
r2ms = np.zeros([len(figtypes), len(keys)])
range_covered = np.zeros([len(figtypes), len(keys)])
range_covered_test = np.zeros([len(figtypes), len(keys)])

for i, key in enumerate(keys):
    fig = plt.figure(constrained_layout=True, figsize=[6, 5])  # time series
    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)

    fig2 = plt.figure(constrained_layout=True, figsize=[8, 3.2])  # Scatter plot
    gs2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig2)

    fig3 = plt.figure(constrained_layout=True, figsize=[8, 3.2])  # histogram
    gs3 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig3)

    ts = [None] * 2
    sc = [None] * 2
    his = [None] * 2
    for bsn in [True, False]:
        model = data[key][behavior][bsn]
        r2ms[1 - bsn, i] = model["R2ms_test"]
        beh = model["signal"] * model["signal_std"] + model["signal_mean"]
        pred = model["output"] * model["signal_std"] + model["signal_mean"]

        range_covered[1 - bsn, i] = frac_range_covered(beh, pred)
        range_covered_test[1 - bsn, i] = frac_range_covered(
            beh[model["test_idx"]], pred[model["test_idx"]]
        )

        # Plot the time series of the prediction and true
        ts[1 - bsn] = fig.add_subplot(gs[1 - bsn, :], sharex=ts[0], sharey=ts[0])
        ts[1 - bsn].plot(model["time"], beh, "k", color=measured_color, lw=1.5)
        ts[1 - bsn].plot(model["time"], pred, color=pred_color, lw=1.5)
        ts[1 - bsn].set_xlabel("Time (s)")
        ts[1 - bsn].set_ylabel(behavior)
        ts[1 - bsn].fill_between(
            [
                model["time"][np.min(model["test_idx"])],
                model["time"][np.max(model["test_idx"])],
            ],
            np.min(beh),
            np.max(beh),
            facecolor=test_color,
            alpha=0.05,
        )
        ts[1 - bsn].set_xticks(np.arange(0, model["time"][-1], 60))
        ts[1 - bsn].axhline(linewidth=0.5, color="k")
        if behavior is "curvature":
            ts[1 - bsn].set_yticks([-2 * np.pi, 0, 2 * np.pi])
            ts[1 - bsn].set_yticklabels([r"$-2\pi$", "0", r"$2\pi$"])

        # plot scatter of prediction vs true
        sc[1 - bsn] = fig2.add_subplot(
            gs2[0, 1 - bsn],
            xlabel="Measured " + behavior,
            ylabel="Predicted " + behavior,
            sharex=sc[0],
            sharey=sc[0],
        )
        sc[1 - bsn].plot(
            beh[model["train_idx"]],
            pred[model["train_idx"]],
            "o",
            color=pred_color,
            label="Train",
            rasterized=False,
            alpha=alpha,
        )
        sc[1 - bsn].plot(
            beh[model["test_idx"]],
            pred[model["test_idx"]],
            "o",
            color=test_color,
            label="Test",
            rasterized=False,
            alpha=alpha_test,
        )
        sc[1 - bsn].plot([min(beh), max(beh)], [min(beh), max(beh)], "k-.")
        sc[1 - bsn].set_title(
            ("BSN" if bsn else "Population")
            + r" $R^2_{\mathrm{ms},\mathrm{test}}$ = %0.3f" % r2ms[1 - bsn, i]
        )
        sc[1 - bsn].legend()
        sc[1 - bsn].set_aspect("equal", adjustable="box")
        if behavior is "curvature":
            sc[1 - bsn].set_yticks([-2 * np.pi, 0, 2 * np.pi])
            sc[1 - bsn].set_yticklabels([r"$-2\pi$", "0", r"$2\pi$"])
            sc[1 - bsn].set_xticks([-2 * np.pi, 0, 2 * np.pi])
            sc[1 - bsn].set_xticklabels([r"$-2\pi$", "0", r"$2\pi$"])

        # plot histogram of predicted and true values for held-out test set
        low_lim = -4
        high_lim = 4
        nbins = 24
        pred_hist, pred_bin_centers, pred_bin_edges = calc_pdf(
            pred[model["test_idx"]], low_lim, high_lim, nbins
        )
        obs_hist, obs_bin_centers, obs_bin_edges = calc_pdf(
            beh[model["test_idx"]], low_lim, high_lim, nbins
        )
        his[1 - bsn] = fig3.add_subplot(
            gs3[0, 1 - bsn],
            xlabel=behavior,
            ylabel="Count",
            title=("BSN" if bsn else "Population")
            + " frac_range_test = %.2f" % range_covered_test[1 - bsn, i],
            sharex=his[0],
            sharey=his[0],
        )
        his[1 - bsn].step(pred_bin_centers, pred_hist, where="mid", label="Prediction")
        his[1 - bsn].step(
            obs_bin_centers, obs_hist, where="mid", label="Observation", color="k"
        )
        his[1 - bsn].set_ylim([0, 1.1 * np.max(np.concatenate([pred_hist, obs_hist]))])
        his[1 - bsn].legend()

    fig.suptitle(key)
    fig2.suptitle(key)
    fig3.suptitle(key)

    pdf.savefig(fig)
    pdf.savefig(fig2)
    pdf.savefig(fig3)

fig4 = plt.figure()
plt.xlabel("Population Performance (Rho2_adj)")
plt.ylabel("Fraction covered")
plt.plot(r2ms[1, :], range_covered_test[1, :], "o", label="Population, Test Set")
plt.plot(r2ms[1, :], range_covered_test[0, :], "+", label="BSN, Test Set")
plt.legend()
pdf.savefig(fig4)

fig4andhalf = plt.figure()
plt.xlabel("Population Performance - BSN performance (Rho2_adj)")
plt.ylabel("Fraction covered")
plt.plot(
    r2ms[1, :] - r2ms[0, :], range_covered_test[1, :], "o", label="Population, Test Set"
)
plt.plot(r2ms[1, :] - r2ms[0, :], range_covered_test[0, :], "+", label="BSN, Test Set")
plt.legend()
pdf.savefig(fig4andhalf)

fig4andthreequarters = plt.figure()
plt.xlabel("Population Performance - BSN performance (Rho2_adj)")
plt.ylabel("POP Fraction covered - BSN FRaction Covered")
POPminusBSN_rho = r2ms[1, :] - r2ms[0, :]
POPminusBSN_range = range_covered_test[1, :] - range_covered_test[0, :]
plt.plot(POPminusBSN_rho, POPminusBSN_range, "o")
ax475 = plt.gca()
ax475.axvline()
ax475.axhline()
plt.legend()
pdf.savefig(fig4andthreequarters)

fig5 = plt.figure(figsize=[2, 5])
plt.title("Test Only")
plt.ylabel("Fraction Covered")
plt.xlabel("0= BSN, 1=Pop")
for k in np.arange(len(range_covered_test[1, :])):
    plt.plot(
        np.array([0, 1]), np.array([range_covered_test[0, k], range_covered_test[1, k]])
    )
plt.xticks([0, 1])
pdf.savefig(fig5)


fig6 = plt.figure()
plt.plot(r2ms[1, :], range_covered[1, :], "o", label="Population, Train + Test")
plt.plot(r2ms[1, :], range_covered[0, :], "+", label="BSN, Train + Test")
plt.legend()
pdf.savefig(fig6)

fig7 = plt.figure(figsize=[2, 5])
plt.title("Train + Test")
plt.ylabel("Fraction Covered")
plt.xticks([0, 1], ["BSN", "Population"])
for k in np.arange(len(range_covered[1, :])):
    plt.plot(np.array([0, 1]), np.array([range_covered[0, k], range_covered[1, k]]))
pdf.savefig(fig7)


pdf.close()
print(("wrote " + outfilename))
