import fastloess as fl
import numpy as np
import matplotlib.pyplot as plt
import polars
from scipy import stats

fig, ax = plt.subplots()
ax.set_ylim(10, 35)
table = polars.read_parquet('/data/cephfs-1/scratch/projects/sawitzki-phenium/transcripts_6.parquet')

x = table['x_location'].to_numpy()[::100]
y = table['y_location'].to_numpy()[::100]
z = table['z_location'].to_numpy()[::100]

model = fl.Loess(fraction=1, iterations=5)
res = model.fit(y, z)

ax.scatter(y, z, s=1)
ax.plot(y, res.y, 'r')
ax.scatter(y, z-res.y + max(res.y), s=1)

res = stats.linregress(y, z-res.y + max(res.y))
print(res.slope)

fig.savefig('/data/cephfs-1/scratch/projects/sawitzki-phenium/y_vs_z_loess.png')
