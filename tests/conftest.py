import matplotlib

# draw figures off-screen so plot tests don't need a display (flaky Tk errors on Windows)
matplotlib.use("Agg")
