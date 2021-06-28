import numpy as np
from numpy.linalg import norm
import pickle
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat, savemat


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
        if i % 100 == 0:
            print(i)
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

with open('circles.pkl', 'rb') as f:
    circles = pickle.load(f)

print(len(circles))
trajectories = find_trajectories(circles)

interp_trajectories = []
for traj in trajectories:
    print('hi')
    ts = traj[:,0]
    xs = traj[:,1]
    ys = traj[:,2]

    xsi = interp1d(ts, xs)
    ysi = interp1d(ts, ys)
    
    interp_trajectories.append([ts[0], ts[-1], xsi, ysi])

pos = [np.array([0, 0])]
for t in range(1, len(lmTime)):
    if t % 100 == 0:
        print(t)
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

xposi = np.interp(hmTime, lmTime + hmTime[0], -(10000./473)*xpos)
yposi = np.interp(hmTime, lmTime + hmTime[0], (10000./473)*ypos)

hrd['dataAll'][0][0][6][:,0] = xposi
hrd['dataAll'][0][0][7][:,0] = yposi

savemat('hiResData_with_pos.mat', hrd)

