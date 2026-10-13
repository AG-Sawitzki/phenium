import numpy as np
import matplotlib.pyplot as plt
import polars
from scipy import stats
from sklearn.ensemble import IsolationForest

fig, ax = plt.subplots()
# ax.set_ylim(10, 35)
table = polars.read_parquet('/data/cephfs-1/scratch/projects/sawitzki-phenium/transcripts_6.parquet')

x = table['x_location'].to_numpy()[::100]
y = table['y_location'].to_numpy()[::100]
z = table['z_location'].to_numpy()[::100]

res = stats.linregress(y, z)

new_z = np.array([])

for fov in set(table['fov_name']):
table_ = table.filter(polars.col('fov_name') == fov)
y_ = table_['y_location'].to_numpy()
z_ = table_['z_location'].to_numpy()
endog = z_ - (res.intercept + res.slope*max(y_))
y_z = np.column_stack([y_, endog])
new_z = np.concatenate([new_z, endog])

y_z = np.column_stack(
    [table['y_location'].to_numpy(), new_z]
)
model = IsolationForest(contamination='auto')
pred = model.fit_predict(y_z)
ax.scatter(table['y_location'].to_numpy(), new_z, s=1, c=pred, cmap='PiYG')


diff = int(max(new_z)-min(new_z))
hist, edges = np.histogram(new_z, bins=diff)
elems = 0
bins = []
bins_by_size = np.argsort(hist)[::-1]
for b in bins_by_size:
bins.append(b)
elems += hist[b]
if elems/len(new_z) >= 0.9:
break

ax.legend()
fig.savefig('/data/cephfs-1/scratch/projects/sawitzki-phenium/y_vs_z_adj_forest.png')


ax.scatter(y, z, s=1)
ax.plot(y, res.intercept + res.slope*y, 'r')
ax.scatter(y, z-(res.slope*y) + max(res.slope*y), s=1)

res = stats.linregress(y, z-(res.slope*y) + max(res.slope*y))
print(res.slope)

fig.savefig('/data/cephfs-1/scratch/projects/sawitzki-phenium/y_vs_z_linreg.png')
