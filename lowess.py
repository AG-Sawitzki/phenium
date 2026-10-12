import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import polars
from scipy import stats

lowess = sm.nonparametric.lowess

fig, ax = plt.subplots()
ax.set_ylim(10, 35)
table = polars.read_parquet('/data/cephfs-1/scratch/projects/sawitzki-phenium/transcripts_6.parquet')

x = table['x_location'].to_numpy()[::100]
y = table['y_location'].to_numpy()[::100]
z = table['z_location'].to_numpy()[::100]

z_lowess = lowess(z, y, frac=0.5, return_sorted=False)

ax.scatter(y, z, s=1)
ax.plot(y, z_lowess, 'r')
ax.scatter(y, z-z_lowess + max(z_lowess), s=1)

res = stats.linregress(y, z-z_lowess + max(z_lowess))
print(res.slope)

fig.savefig('/data/cephfs-1/scratch/projects/sawitzki-phenium/y_vs_z_lowess.png')
