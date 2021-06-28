import numpy as np
from numpy.linalg import norm
import pickle
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat


hrd = loadmat('hiResData.mat')
hmTime = hrd['dataAll'][0][0][1][:,0]
stage_xpos = hrd['dataAll'][0][0][6][:,0]
stage_ypos = hrd['dataAll'][0][0][7][:,0]
lmTime = loadmat('heatDataMS.mat')['clTime'][:,0]

def find_trajectories(pts, min_length = 2, max_vel = 5, max_jump = 10):   #min_length is number of frames the trajectory is in, max_vel is the biggest pixel/frame jump we accept
    trajectories = []
    for p in pts[0]:
        trajectories.append([p])

    for i in range(1, len(pts)):
        for p in pts[i]:
            dists = [] 
            candidates = filter(lambda x: x[-1][0] >= p[0] - max_jump, trajectories)
            for t in candidates:
                x1 = np.array([t[-1][1], t[-1][2]])
                x2 = np.array([p[1], p[2]])
                s = np.linalg.norm(x2-x1)/(p[0]-t[-1][0])
                # dangles.append(np.abs(np.arctan2(t[-1][2],t[-1][1])-np.arctan2(p[2],p[1])))
                dists.append(np.linalg.norm(x1-x2))
            if not dists:
                trajectories.append([p])
                continue
            t_idx = np.argmin(np.array(dists))
            t = candidates[t_idx]
            x1 = np.array([t[-1][1], t[-1][2]])
            x2 = np.array([p[1], p[2]])
            s = np.linalg.norm(x2-x1)/(p[0]-t[-1][0])
            if s < max_vel:
                t.append(p)
            else:
                trajectories.append([p])
    trajectories = list(map(lambda x: np.array(x), filter(lambda x: len(x) > min_length, trajectories)))
    return trajectories

with open('circles_control.pkl', 'rb') as f:
    circles = pickle.load(f)

print(len(circles))
trajectories = find_trajectories(circles)

interp_trajectories = []
for traj in trajectories:
    ts = traj[:,0]
    xs = traj[:,1]
    ys = traj[:,2]

    xsi = interp1d(ts, xs)
    ysi = interp1d(ts, ys)
    
    interp_trajectories.append([ts[0], ts[-1], xsi, ysi])

pos = [np.array([0, 0])]
for t in range(1, len(lmTime)):
    vel_ests = []
    for traj in filter(lambda x: x[0] < t <= x[1], interp_trajectories):
        vel_ests.append([traj[2](t) - traj[2](t-1), traj[3](t) - traj[3](t-1)])
    
    if vel_ests:
        pos.append(pos[-1] + np.mean(np.array(vel_ests), axis = 0))
    else:
        pos.append(pos[-1])

pos = np.array(pos)
xpos = pos[:,0]
ypos = pos[:,1]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize = (20, 15))

ax[0,0].set_xlabel('Time', fontsize = 20)
ax[1,0].set_xlabel('Time', fontsize = 20)
ax[0,0].set_ylabel('X Position', fontsize = 20)
ax[1,0].set_ylabel('Y Position', fontsize = 20)

ax[0,0].set_xlim(50, 300)
ax[0,0].set_ylim(-4000, 4000)
ax[0,0].plot(lmTime + hmTime[0], -(10000./473)*xpos + stage_xpos[3], label = 'Inferred')
ax[0,0].plot(hmTime, stage_xpos, label = 'Actual')
ax[0,0].legend()

ax[1,0].set_xlim(50, 300)
ax[1,0].set_ylim(-4000, 4000)
ax[1,0].plot(lmTime + hmTime[0], (10000./473)*ypos + stage_ypos[3], label = 'Inferred')
ax[1,0].plot(hmTime, stage_ypos, label = 'Actual')
ax[1,0].legend()

xvel = -50*gaussian_filter1d(xpos, order = 1, sigma = 50)
stage_xvel = -200*gaussian_filter1d(stage_xpos, order = 1, sigma = 200)
yvel = -50*gaussian_filter1d(ypos, order = 1, sigma = 50)
stage_yvel = -200*gaussian_filter1d(stage_ypos, order = 1, sigma = 200)

xveli = np.interp(hmTime, lmTime + hmTime[0], -(10000./473)*xvel)
yveli = np.interp(hmTime, lmTime + hmTime[0], (10000./473)*yvel)

xrho = np.corrcoef(xveli, stage_xvel)[0,1]
yrho = np.corrcoef(yveli, stage_yvel)[0,1]

ax[0,1].set_xlabel('Time', fontsize = 20)
ax[1,1].set_xlabel('Time', fontsize = 20)
ax[0,1].set_ylabel('X Velocity', fontsize = 20)
ax[1,1].set_ylabel('Y Velocity', fontsize = 20)

ax[0,1].set_xlim(50, 300)
ax[0,1].plot(lmTime + hmTime[0], -(10000./473)*xvel, label = 'Inferred')
ax[0,1].plot(hmTime, stage_xvel, label = 'Actual')
ax[0,1].set_title(r'$\rho = %0.2f$' % xrho, fontsize = 24)
ax[0,1].legend()

ax[1,1].set_xlim(50, 300)
ax[1,1].plot(lmTime + hmTime[0], (10000./473)*yvel, label = 'Inferred')
ax[1,1].plot(hmTime, stage_yvel, label = 'Actual')
ax[1,1].set_title(r'$\rho = %0.2f$' % yrho, fontsize = 24)
ax[1,1].legend()

fig.savefig('recovery.png')

