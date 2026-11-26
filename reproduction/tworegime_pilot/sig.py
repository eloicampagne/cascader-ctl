import sys, pandas as pd, numpy as np
from scipy.stats import ttest_rel
def cells(regime):
    df=pd.read_csv(f'pilot_{regime}_raw.csv'); out=[]
    for (s,b),g in df.groupby(['scatter','b']):
        ser={}
        for m in ['Chain','CTL-MST','CTL-Forest']:
            t=g[(g.split=='tune')&(g.method==m)].groupby('eta')['value'].mean().dropna()
            e=t.idxmin(); ser[m]=g[(g.split=='eval')&(g.method==m)&(g.eta==e)].set_index('rep')['value']
        p=pd.concat(ser,axis=1).dropna()
        for m in ['CTL-MST','CTL-Forest']:
            gap=(p[m]-p['Chain'])/p['Chain']*100
            out.append(dict(regime=regime,scatter=s,b=b,method=m,gap=gap.mean(),se=gap.std(ddof=1)/np.sqrt(len(gap)),
                            p=ttest_rel(p[m],p['Chain']).pvalue,n=len(gap)))
    return pd.DataFrame(out)
if __name__=='__main__':
    o=pd.concat([cells(r) for r in ['syn','old','syn_s05','syn_n512']]); o.to_csv('cells_sig.csv',index=False)
    s=o[o.regime=='syn']
    for m in ['CTL-MST','CTL-Forest']:
        g=s[s.method==m]; print(m,'lower:',(g.gap<0).sum(),'lower&p<.05:',((g.gap<0)&(g.p<.05)).sum(),'higher&p<.05:',((g.gap>0)&(g.p<.05)).sum(), 'max gap', g.gap.max().round(2))
        print(g.assign(sig=lambda d:np.where((d.gap<0)&(d.p<.05),'*','')).pivot_table(index='scatter',columns='b',values='sig',aggfunc='first').to_string())
