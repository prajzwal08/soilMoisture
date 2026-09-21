import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,'/gpfs/work3/0/prjs1968/soilMoisture')
import dataset as _ds; _ds.ZARR_ROOT = Path('/projects/prjs1968/zarr_tokens')
from dataset import _load_zarr_labels, _open_zarr
SIX=['CR200-18','CR200-25','CR1000-2','CR200-24','CR200-15','CR200-6']
b=pd.read_csv('csvs/ecostress_dtr_bundles.TxSON.csv').set_index('station_id')
D={}; S={}
for s in SIX:
    r=b.loc[s]; z=np.load(r['path'],allow_pickle=False)
    ok=(z['grid_aligned']==1)&(z['n_valid_px']>0); nd=int(ok.sum())
    dtr=z['dtr_k'][ok].reshape(nd,-1); val=z['valid'][ok].reshape(nd,-1).astype(bool)
    with np.errstate(invalid='ignore'):
        sc=np.array([d[v].mean() if v.any() else np.nan for d,v in zip(dtr,val)])
    dt=pd.to_datetime([str(x.decode() if isinstance(x,bytes) else x)[:10] for x in z['day_utc'][ok]]).normalize()
    D[s]=pd.Series(sc,index=dt).dropna()
    zg=_open_zarr(Path('/projects/prjs1968/zarr_tokens')/r['category']/r['folder'],r['category'])
    sm,dep,tm,qc=_load_zarr_labels(zg)
    for i,dd in enumerate(dep):
        dd=dd.decode() if isinstance(dd,bytes) else str(dd)
        if dd=='0-10':
            k=~np.isnan(sm[i]);  k&= (qc[i]==0) if qc is not None else True
            S[s]=pd.Series(sm[i][k],index=pd.to_datetime(tm[k]).normalize())
dtr=pd.DataFrame(D); smf=pd.DataFrame(S)
common=dtr.dropna().index
print(f"dates where all six have a usable DTR: {len(common)}")
d6=dtr.loc[common]; s6=smf.reindex(common)[SIX]
print("\nBETWEEN-STATION spread on those dates (six probes, 0.4-0.94 km apart):")
print(f"  DTR : mean across-station SD = {d6.std(axis=1).mean():.3f} K   "
      f"(mean level {d6.values.mean():.1f} K)  -> CV {100*d6.std(axis=1).mean()/abs(d6.values.mean()):.1f}%")
print(f"  SM  : mean across-station SD = {s6.std(axis=1).mean():.4f} m3/m3 "
      f"(mean level {np.nanmean(s6.values):.3f})  -> CV {100*s6.std(axis=1).mean()/np.nanmean(s6.values):.1f}%")
print("\npairwise correlation of the six DTR time series:")
c=d6.corr().values; iu=np.triu_indices(6,1)
print(f"  mean r = {c[iu].mean():+.4f}   min {c[iu].min():+.4f}")
print("\npairwise correlation of the six SM series on the same dates:")
c2=s6.corr().values
print(f"  mean r = {c2[iu].mean():+.4f}   min {c2[iu].min():+.4f}")
print("\nstation mean SM vs station mean DTR (between-station, n=6):")
g=pd.DataFrame({'sm':s6.mean(),'dtr':d6.mean()}).dropna()
print(g.round(3).to_string())
print(f"  r = {np.corrcoef(g.sm,g.dtr)[0,1]:+.3f}")
