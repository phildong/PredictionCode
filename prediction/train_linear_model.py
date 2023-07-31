import pickle

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

from .models import linear

for gtype in ["gcamp", "gfp"]:
    with open("intermediate/%s_recordings.dat" % gtype, "rb") as f:
        data = pickle.load(f)

    results = {}

    for key in list(data.keys()):
        # book-keeping
        print(("Running " + key))
        time = data[key]["time"]
        neurons = data[key]["neurons"]
        nderiv = data[key]["neuron_derivatives"]
        velocity = data[key]["velocity"]
        curvature = data[key]["curvature"]
        results[key] = {}
        results[key]["velocity"] = {}
        results[key]["curvature"] = {}
        # generate inputs
        pca = PCA(whiten=True)
        neuron_pca = pca.fit_transform(neurons.T).T
        neuron_pca_first = neuron_pca[:3, :]
        neuron_pca_rd = neuron_pca[3:, :]
        pca = PCA(whiten=True)
        nderiv_pca = pca.fit_transform(nderiv.T).T
        nderiv_pca_first = nderiv_pca[:3, :]
        nderiv_pca_rd = nderiv_pca[3:, :]
        nnd_pca = np.vstack((neuron_pca, nderiv_pca))
        nnd_pca_first = np.vstack((neuron_pca_first, nderiv_pca_first))
        nnd_pca_rd = np.vstack((neuron_pca_rd, nderiv_pca_rd))
        neurons_and_derivs = np.vstack((neurons, nderiv))
        # run pcr
        for pcr, Xin in {
            "full": nnd_pca,
            "first": nnd_pca_first,
            "reduced": nnd_pca_rd,
        }.items():
            results[key]["velocity"][pcr] = linear.optimize(
                time,
                Xin,
                velocity,
                options_override={
                    "l1_ratios": [0],
                    "parallelize": False,
                    "best_neuron": False,
                },
            )
            results[key]["curvature"][pcr] = linear.optimize(
                time,
                Xin,
                curvature,
                options_override={
                    "l1_ratios": [0],
                    "parallelize": False,
                    "best_neuron": False,
                },
            )
            print(
                (
                    "\tVelocity (%s)  R^2_ms = %0.2f"
                    % (pcr, results[key]["velocity"][pcr]["R2ms_test"])
                )
            )
            print(
                (
                    "\tCurvature (%s) R^2_ms = %0.2f"
                    % (pcr, results[key]["curvature"][pcr]["R2ms_test"])
                )
            )

        for bsn in [True, False]:
            results[key]["velocity"][bsn] = linear.optimize(
                time,
                neurons_and_derivs,
                velocity,
                options_override={
                    "l1_ratios": [0],
                    "parallelize": False,
                    "best_neuron": bsn,
                },
            )
            results[key]["curvature"][bsn] = linear.optimize(
                time,
                neurons_and_derivs,
                curvature,
                options_override={
                    "l1_ratios": [0],
                    "parallelize": False,
                    "best_neuron": bsn,
                },
            )
            print(
                (
                    "\tVelocity %s  R^2_ms = %0.2f"
                    % (
                        "(BSN):       " if bsn else "(Population):",
                        results[key]["velocity"][bsn]["R2ms_test"],
                    )
                )
            )
            print(
                (
                    "\tCurvature %s R^2_ms = %0.2f"
                    % (
                        "(BSN):       " if bsn else "(Population):",
                        results[key]["curvature"][bsn]["R2ms_test"],
                    )
                )
            )

    with open("intermediate/%s_linear_models.dat" % gtype, "wb") as f:
        pickle.dump(results, f)
