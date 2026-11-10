"""Step 3b. Top-three-driver sensitivity check (independent of decision.py; needs only out/fitted.pkl)."""
from common import *
import joblib, time
from sklearn.base import clone
F=joblib.load(OUT / 'fitted.pkl')
pooled_proto=F[('Pooled','LogisticRegression')][0]
stage_proto={s:F[(s,'LogisticRegression')][0] for s in ST}
Xp,yp,_,_=vdata('Pooled')
FLAG=0.10
def raw_of(n):
    rest=n[len('cat__'):] if n.startswith('cat__') else None
    if n.startswith('num__'): return n[len('num__'):]
    return next((c for c in CAT if rest.startswith(c)),rest)
def lin_shap(pipe,X,bg):
    pr,clf=pipe.named_steps['prep'],pipe.named_steps['clf']
    T=pr.transform(X); T=T.toarray() if hasattr(T,'toarray') else T
    B=pr.transform(bg); B=B.toarray() if hasattr(B,'toarray') else B
    phi=np.abs(clf.coef_[0][None,:]*(T-B.mean(0)))
    names=[raw_of(n) for n in pr.get_feature_names_out()]
    out={}
    for n,col in zip(names,phi.T): out[n]=out.get(n,0)+col
    keys=sorted(out); return np.column_stack([out[k] for k in keys]),keys
def top1(pipe,X,bg):
    M,keys=lin_shap(pipe,X,bg); return np.array(keys)[M.argmax(1)]
def flags(p,share=FLAG):
    r=pd.Series(p).rank(pct=True).values; return r>(1-share)
def boot_fit(proto,X,y,rng):
    pos,neg=np.where(y.values==1)[0],np.where(y.values==0)[0]
    ii=np.concatenate([rng.choice(pos,len(pos)),rng.choice(neg,len(neg))])
    return clone(proto).fit(X.iloc[ii],y.iloc[ii])

def top3(pipe,X,bg):
    M,keys=lin_shap(pipe,X,bg); k=np.array(keys); return np.argsort(-M,axis=1)[:,:3]
def ag3(a,b): return np.mean([len(set(x)&set(y))/3 for x,y in zip(a,b)])
rng=np.random.default_rng(RANDOM_STATE+7)
B=200; data={s:vdata(s) for s in ST}
obs={s:ag3(top3(pooled_proto,data[s][0],data[s][0]),top3(stage_proto[s],data[s][0],data[s][0])) for s in ST}
R={s:{'ss':[],'pp':[],'ps':[]} for s in ST}
for b in range(B):
    p1,p2=boot_fit(pooled_proto,Xp,yp,rng),boot_fit(pooled_proto,Xp,yp,rng)
    for s in ST:
        Xtr,ytr=data[s][:2]
        s1,s2=boot_fit(stage_proto[s],Xtr,ytr,rng),boot_fit(stage_proto[s],Xtr,ytr,rng)
        T={k:top3(m,Xtr,Xtr) for k,m in (('p1',p1),('p2',p2),('s1',s1),('s2',s2))}
        R[s]['ss'].append(ag3(T['s1'],T['s2'])); R[s]['pp'].append(ag3(T['p1'],T['p2'])); R[s]['ps'].append(ag3(T['s1'],T['p1']))
    if b%20==0: print('refit',b,flush=True)
joblib.dump(dict(obs=obs,R=R),OUT / 'top3_out.pkl')
pct=lambda v:(np.mean(v)*100,*np.percentile(v,[2.5,97.5])*100)
for s in ST:
    gap=np.array(R[s]['ss'])-np.array(R[s]['ps'])
    print(f"{s:10} observed {obs[s]*100:.1f} | two stage refits %.1f [%.1f,%.1f] | refit pooled-vs-stage %.1f [%.1f,%.1f] | gap %.1f [%.1f,%.1f]"%(*pct(R[s]['ss']),*pct(R[s]['ps']),*pct(gap)))
G=np.mean([np.array(R[s]['ss'])-np.array(R[s]['ps']) for s in ST],axis=0)
print('POOLED top3 gap %.1f [%.1f,%.1f]'%pct(G))
