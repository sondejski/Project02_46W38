# turbie_mod.py
# super simple 2-dof "turbie" model.
# sorry if code looks messy. i just want it to run 

import numpy as np

PAR = None
CTTAB = None

def load_params(path_txt):
    # reads numbers in fixed order from the file
    global PAR
    vals = []
    for line in open(path_txt, "r"):
        s = line.strip()
        if len(s)==0 or s.startswith("#"):
            continue
        try:
            vals.append(float(s.split("#")[0].strip()))
        except:
            pass
    PAR = {
        "mb": vals[0],   # one blade
        "mn": vals[1],   # nacelle
        "mh": vals[2],   # hub
        "mt": vals[3],   # tower
        "c1": vals[4],
        "c2": vals[5],
        "k1": vals[6],
        "k2": vals[7],
        "fb": vals[8],3
        "ft": vals[9],
        "drb": vals[10],
        "drt": vals[11],
        "Dr": vals[12],  # rotor diameter
        "rho": vals[13], # air density
    }
    return PAR

def load_CT_table(path_txt):
    global CTTAB
    v, ct = [], []
    for line in open(path_txt, "r"):
        s = line.strip()
        if len(s)==0 or s.startswith("#"):
            continue
        p = s.split()
        if len(p)>=2:
            try:
                v.append(float(p[0])); ct.append(float(p[1]))
            except:
                pass
    CTTAB = (np.array(v, float), np.array(ct, float))
    return CTTAB

def ct_from_mean(mean_ws):
    v, ct = CTTAB
    if mean_ws <= v[0]: return float(ct[0])
    if mean_ws >= v[-1]: return float(ct[-1])
    i = np.searchsorted(v, mean_ws) - 1
    x0,x1 = v[i], v[i+1]
    y0,y1 = ct[i], ct[i+1]
    a = (mean_ws - x0)/(x1 - x0)
    return float(y0 + a*(y1 - y0))

def mats():
    m1 = 3.0*PAR["mb"]  # 3 blades
    m2 = PAR["mn"] + PAR["mh"] + PAR["mt"]
    M = np.array([[m1,0.0],[0.0,m2]], float)
    C = np.array([[PAR["c1"],0.0],[0.0,PAR["c2"]]], float)
    K = np.array([[PAR["k1"],0.0],[0.0,PAR["k2"]]], float)
    return M,C,K

def force_from_wind(u, ct_const):
    R = 0.5*PAR["Dr"]
    A = np.pi*R*R
    F1 = 0.5*PAR["rho"]*A*ct_const*(u*u)
    return np.array([F1, 0.0], float)  # only blade dof

def rhs(t, y, M, C, K, tw, uw, ct_const):
    # y = [xb, xt, vb, vt]
    # tiny linear interp for wind u(t)
    if t <= tw[0]:
        u = uw[0]
    elif t >= tw[-1]:
        u = uw[-1]
    else:
        j = np.searchsorted(tw, t) - 1
        t0,t1 = tw[j], tw[j+1]
        u0,u1 = uw[j], uw[j+1]
        if t1 == t0:
            u = u0
        else:
            a = (t - t0)/(t1 - t0)
            u = u0 + a*(u1 - u0)

    x = y[0:2]; v = y[2:4]
    F = force_from_wind(u, ct_const)
    Minv = np.linalg.inv(M)   
    a = Minv.dot(F - C.dot(v) - K.dot(x))

    dy = np.zeros_like(y)
    dy[0:2] = v
    dy[2:4] = a
    return dy

def rk4(fun, t, y, h):
    k1 = fun(t, y)
    k2 = fun(t + 0.5*h, y + 0.5*h*k1)
    k3 = fun(t + 0.5*h, y + 0.5*h*k2)
    k4 = fun(t + h, y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate(tw, uw, y0):
    # very tiny integrator that uses wind dt as step
    M,C,K = mats()
    ct_const = ct_from_mean(float(np.mean(uw)))

    if len(tw)>1:
        h = float(tw[1]-tw[0])
    else:
        h = 0.1

    N = int((tw[-1]-tw[0])/h) + 1
    T = np.zeros(N, float)
    Y = np.zeros((N, len(y0)), float)
    T[0] = float(tw[0])
    Y[0,:] = y0

    def f(tt, yy): return rhs(tt, yy, M, C, K, tw, uw, ct_const)

    for i in range(1, N):
        T[i] = T[i-1] + h
        Y[i,:] = rk4(f, T[i-1], Y[i-1,:], h)

    return T, Y, ct_const
