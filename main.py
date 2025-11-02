# main.py
# super simple runner. reads inputs, runs model, saves txt, draws 1 plot.
# sorry for globals and simple code. i just want it to work :)

import os
import numpy as np
import matplotlib.pyplot as plt
import turbie_mod as tm

# folders next to this file
HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "inputs")
WIND = os.path.join(INP, "wind_files")
TURB = os.path.join(INP, "turbie_inputs")
OUT = os.path.join(HERE, "outputs")

if not os.path.exists(OUT):
    os.makedirs(OUT, exist_ok=True)

# load params and CT
tm.load_params(os.path.join(TURB, "turbie_parameters.txt"))
tm.load_CT_table(os.path.join(TURB, "CT.txt"))

def read_wind(path_txt):
    # expects 2 columns: time  windspeed
    t = []
    u = []
    for line in open(path_txt, "r"):
        s = line.strip()
        if len(s)==0 or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts)>=2:
            try:
                t.append(float(parts[0]))
                u.append(float(parts[1]))
            except:
                pass
    return np.array(t, float), np.array(u, float)

# collect all .txt winds
winds = []
if os.path.isdir(WIND):
    for name in sorted(os.listdir(WIND)):
        if name.lower().endswith(".txt"):
            winds.append(os.path.join(WIND, name))

# if no wind files, make a tiny demo
if len(winds)==0:
    print("no wind files found, so i made a tiny demo wind.")
    t = np.linspace(0, 60, 601)
    u = 8.0 + 0.5*np.sin(2*np.pi*t/10.0)
    winds = [None]  # marker to use demo
else:
    print("found", len(winds), "wind files")

# run each case
first_plot_done = False
for wf in winds:
    if wf is None:
        t, u = t, u  # reuse demo made above
        tag = "demo"
    else:
        t, u = read_wind(wf)
        tag = os.path.splitext(os.path.basename(wf))[0]

    y0 = np.array([0.0, 0.0, 0.0, 0.0])  # [xb, xt, vb, vt]
    T, Y, used_ct = tm.simulate(t, u, y0)

    # means and stds (displacements only)
    xb = Y[:,0]
    xt = Y[:,1]
    mxb = float(np.mean(xb))
    mxt = float(np.mean(xt))
    sxb = float(np.std(xb))
    sxt = float(np.std(xt))

    # save timeseries
    out_ts = os.path.join(OUT, f"{tag}_displacements.txt")
    with open(out_ts, "w") as f:
        f.write("# t  xb  xt  vb  vt\n")
        for i in range(len(T)):
            f.write(f"{T[i]} {Y[i,0]} {Y[i,1]} {Y[i,2]} {Y[i,3]}\n")
    print("saved", out_ts)

    # save simple stats
    out_st = os.path.join(OUT, f"{tag}_mean_std.txt")
    with open(out_st, "w") as f:
        f.write("# mean_xb  std_xb  mean_xt  std_xt  CT_used\n")
        f.write(f"{mxb} {sxb} {mxt} {sxt} {used_ct}\n")
    print("saved", out_st)

    # one simple plot for the first case only
    if not first_plot_done:
        plt.figure()
        plt.plot(t, u, label="wind U(t)")
        plt.plot(T, xb, label="blade x(t)")
        plt.plot(T, xt, label="tower x(t)")
        plt.xlabel("time [s]")
        plt.ylabel("values")
        plt.title(f"{tag}  (CT={used_ct:.3f})")
        plt.legend()
        figp = os.path.join(OUT, f"{tag}_plot.png")
        plt.savefig(figp, dpi=140, bbox_inches="tight")
        print("saved", figp)
        first_plot_done = True

print("done.")
