from .models import linear, mars
from .train_linear_model import dFdt
import pickle

with open('gcamp_recordings.dat', 'rb') as f:
    data = pickle.load(f)
    
results = {}

for key in list(data.keys()):
    print(("Running "+key))
    time = data[key]['time']
    neurons = data[key]['neurons']
    velocity = data[key]['velocity']
    curvature = data[key]['curvature']

    nderiv = dFdt(neurons)
    neurons_and_derivs = np.vstack((neurons, nderiv))

    results[key] = {}
    results[key]['velocity'] = {}
    results[key]['curvature'] = {}

    print("\t main")
    results[key]['velocity']['main'] = linear.optimize(time, neurons_and_derivs, velocity, options = {"l1_ratios": [0], "parallelize": False})
    results[key]['curvature']['main'] = linear.optimize(time, neurons_and_derivs, curvature, options = {"l1_ratios": [0], "parallelize": False})
    print((results[key]['velocity']['main']['scorespredicted'][1], results[key]['curvature']['main']['scorespredicted'][1]))

    print("\t no_deriv")
    results[key]['velocity']['no_deriv'] = linear.optimize(time, neurons, velocity, options = {"l1_ratios": [0], "parallelize": False})
    results[key]['curvature']['no_deriv'] = linear.optimize(time, neurons, curvature, options = {"l1_ratios": [0], "parallelize": False})
    print((results[key]['velocity']['no_deriv']['scorespredicted'][1], results[key]['curvature']['no_deriv']['scorespredicted'][1]))

    print("\t acc")
    results[key]['velocity']['acc'] = linear.optimize(time, neurons_and_derivs, velocity, options = {"l1_ratios": [0], "parallelize": False, "derivative_penalty": True})
    results[key]['curvature']['acc'] = linear.optimize(time, neurons_and_derivs, curvature, options = {"l1_ratios": [0], "parallelize": False, "derivative_penalty": True})
    print((results[key]['velocity']['acc']['scorespredicted'][1], results[key]['curvature']['acc']['scorespredicted'][1]))

    print("\t no_deriv_acc")
    results[key]['velocity']['no_deriv_acc'] = linear.optimize(time, neurons, velocity, options = {"l1_ratios": [0], "parallelize": False, "derivative_penalty": True})
    results[key]['curvature']['no_deriv_acc'] = linear.optimize(time, neurons, curvature, options = {"l1_ratios": [0], "parallelize": False, "derivative_penalty": True})
    print((results[key]['velocity']['no_deriv_acc']['scorespredicted'][1], results[key]['curvature']['no_deriv_acc']['scorespredicted'][1]))

    print("\t l0.01")
    results[key]['velocity']['l0.01'] = linear.optimize(time, neurons_and_derivs, velocity, options = {"l1_ratios": [0.01], "parallelize": False})
    results[key]['curvature']['l0.01'] = linear.optimize(time, neurons_and_derivs, curvature, options = {"l1_ratios": [0.01], "parallelize": False})
    print((results[key]['velocity']['l0.01']['scorespredicted'][1], results[key]['curvature']['l0.01']['scorespredicted'][1]))

    print("\t no_deriv_l0.01")
    results[key]['velocity']['no_deriv_l0.01'] = linear.optimize(time, neurons, velocity, options = {"l1_ratios": [0.01], "parallelize": False})
    results[key]['curvature']['no_deriv_l0.01'] = linear.optimize(time, neurons, curvature, options = {"l1_ratios": [0.01], "parallelize": False})
    print((results[key]['velocity']['no_deriv_l0.01']['scorespredicted'][1], results[key]['curvature']['no_deriv_l0.01']['scorespredicted'][1]))

    print("\t tree")
    results[key]['velocity']['tree'] = linear.optimize(time, neurons, velocity, options = {"l1_ratios": [0], "decision_tree": True, "parallelize": False})
    results[key]['curvature']['tree'] = linear.optimize(time, neurons, curvature, options = {"l1_ratios": [0], "decision_tree": True, "parallelize": False})
    # print(results[key]['velocity']['tree']['scorespredicted'][1], results[key]['curvature']['tree']['scorespredicted'][1])

    print("\t mars")
    results[key]['velocity']['mars'] = mars.optimize(time, neurons, velocity)
    results[key]['curvature']['mars'] = mars.optimize(time, neurons, curvature)
    print((results[key]['velocity']['mars']['scorespredicted'][1], results[key]['curvature']['mars']['scorespredicted'][1]))
    
with open('all_trained_models.dat', 'wb') as f:
    pickle.dump(results, f)