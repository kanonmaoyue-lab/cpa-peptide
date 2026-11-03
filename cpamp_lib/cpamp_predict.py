"""Version-aware dual-channel inference; defaults are selected by paired evaluation."""
from pathlib import Path
import sys,json,pickle,argparse
ROOT=Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT.parent/'.venv/Lib/site-packages'))
import numpy as np
import pandas as pd
from cpamp_sequence import normalize_sequence,replace_sequence_block
from cpamp_retrain import load_script,corrected_delta,OUT,TASKS,TASK_COLS

def build_features(smiles,sequences):
    """Return feature variants plus original valid row positions, without row loss."""
    from rdkit import Chem,RDLogger
    RDLogger.DisableLog('rdApp.*')
    smiles=list(smiles);sequences=list(sequences)
    if len(smiles)!=len(sequences):raise ValueError('SMILES and sequence row counts differ')
    keep=np.array([i for i,s in enumerate(smiles) if isinstance(s,str) and Chem.MolFromSmiles(s) is not None],int)
    if not len(keep):return {k:np.empty((0,2331),np.float32) for k in ['legacy','fixed','fixed_delta']},keep,[]
    ss=[smiles[i] for i in keep];seq=[normalize_sequence(sequences[i]) for i in keep]
    m90=load_script('90_build_expanded.py')
    old,retained=m90.compute_features(ss,['']*len(ss))
    if not np.array_equal(retained,np.arange(len(ss))):raise ValueError('Molecular featurizer unexpectedly removed a parsed structure')
    fixed=replace_sequence_block(old,seq)
    names=np.load(ROOT/'datasets/DBAASP_full/expanded.npz',allow_pickle=True)['names']
    delta,_=corrected_delta(fixed,ss,seq,names)
    return {'legacy':old,'fixed':fixed,'fixed_delta':delta},keep,seq

def estimators(obj):
    return [v[2] if isinstance(v,tuple) else v for v in obj['models']]

def load_model(channel,mode='active'):
    if mode=='active':
        path=OUT/'active_models.json'
        if not path.exists():raise RuntimeError('Model selection is not yet finalized; use an explicit archived candidate')
        registry=json.loads(path.read_text(encoding='utf8'))[channel]
        model_path=ROOT/registry['path'];variant=registry['feature_candidate']
    elif mode=='original':
        model_path=ROOT/('artifacts/pantarget/pantarget_models.pkl' if channel=='qual' else 'artifacts/expanded/expanded_models.pkl')
        variant='legacy'
    else:
        model_path=OUT/f'model_{channel}_{mode}.pkl';variant=mode
    return pickle.loads(model_path.read_bytes()),variant,str(model_path.relative_to(ROOT))

def max_similarity(A,B,batch=64):
    """Exact all-reference Morgan Tanimoto; no sampling and no integer overflow."""
    A=np.asarray(A,np.float32);B=np.asarray(B,np.float32)
    bsum=B.sum(axis=1);out=[]
    for start in range(0,len(A),batch):
        part=A[start:start+batch];both=part@B.T
        union=part.sum(axis=1)[:,None]+bsum[None,:]-both
        sim=np.divide(both,union,out=np.zeros_like(both),where=union>0)
        out.extend(sim.max(axis=1))
    return np.array(out)

def predict_dataframe(frame,strain=None,mode='active'):
    if 'SMILES' not in frame:raise ValueError('Input requires a SMILES column')
    data=frame.copy().reset_index(drop=True)
    if 'ID' not in data:data['ID']=[f'P{i+1}' for i in range(len(data))]
    sequences=data.Seq.fillna('').tolist() if 'Seq' in data else ['']*len(data)
    features,keep,resolved=build_features(data.SMILES.fillna('').tolist(),sequences)
    result=data.copy();result['状态']='SMILES 无法解析'
    if not len(keep):return result
    result.loc[keep,'状态']='成功'
    result.loc[keep,'有序列']=['是' if s else '否' for s in resolved]
    result.loc[keep,'序列字符数']=[len(s) for s in resolved]
    result.loc[keep,'序列提示']=['缺失：部分特征保留缺失' if not s else '含X：未知残基未猜测' if 'X' in s else '母体残基投影；不表示修饰化学' for s in resolved]
    qual,qvariant,qpath=load_model('qual',mode)
    key='fixed' if qvariant=='regularized' else qvariant
    X=features[key][:,:2277]
    P=[]
    for mdl in estimators(qual):
        branch=[]
        for j in range(5):
            target=np.zeros((len(X),5),np.float32);target[:,j]=1
            branch.append(mdl.predict_proba(np.hstack([X,target]))[:,1])
        P.append(np.column_stack(branch))
    P=np.mean(P,axis=0)
    display=['总抗菌','抗细菌','抗革兰阳性','抗革兰阴性','抗真菌']
    for j,name in enumerate(display):result.loc[keep,'定性_'+name]=P[:,j]
    result.loc[keep,'定性模型']=qpath
    result.loc[keep,'定性特征版本']=qvariant
    qtrain=np.load(OUT/'qual_train.npz',allow_pickle=True)['X_legacy'][:,62:2110]
    qsim=max_similarity(features['legacy'][:,62:2110],qtrain)
    result.loc[keep,'定性_最大训练相似度']=qsim
    quant,rvariant,rpath=load_model('quant',mode)
    key='fixed' if rvariant=='regularized' else rvariant
    X=features[key];targets=[str(data.loc[i,'Strain']) if 'Strain' in data and pd.notna(data.loc[i,'Strain']) else str(strain or '') for i in keep]
    group=quant.get('strain_groups',quant.get('strain_group',{}));codes=quant['strain_codes'];meds=quant['strain_median']
    order=['gram_pos','gram_neg','fungal','other'];C=np.zeros((len(keep),6),np.float32)
    for i,t in enumerate(targets):
        g=group.get(t,'other');C[i,order.index(g) if g in order else 3]=1
        C[i,4]=codes.get(t,0);C[i,5]=meds.get(t,0.)
    pred=np.mean([mdl.predict(np.hstack([X,C])) for mdl in estimators(quant)],axis=0)
    known=np.array([bool(t) and t in codes for t in targets])
    specified=np.array([bool(t) for t in targets])
    result.loc[keep,'目标菌株']=targets
    result.loc[keep,'定量_log10uM']=np.where(specified,pred,np.nan)
    result.loc[keep,'定量_uM']=np.where(specified,np.power(10.,pred),np.nan)
    result.loc[keep,'菌株已训练']=known
    result.loc[keep,'定量模型']=rpath
    result.loc[keep,'定量特征版本']=rvariant
    rtrain=np.load(OUT/'quant_train.npz',allow_pickle=True)['X_legacy'][:,62:2110]
    rsim=max_similarity(features['legacy'][:,62:2110],rtrain)
    result.loc[keep,'定量_最大训练相似度']=rsim
    result.loc[keep,'外推提示']=[
        '未指定目标菌株：仅输出定性' if not specified[i] else
        '新菌株且结构远离训练域：定量外推证据不足' if not known[i] and rsim[i]<.4 else
        '新菌株：数值仅作研究性参考' if not known[i] else
        '结构远离训练域：建议补充实验锚点' if rsim[i]<.4 else
        '结构覆盖有限：谨慎外推' if rsim[i]<.7 else '处于较近的结构覆盖范围'
        for i in range(len(keep))]
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--strain',default='');ap.add_argument('--model',default='active',choices=['active','original','legacy','fixed'])
    args=ap.parse_args()
    inp=Path(args.input);frame=pd.read_excel(inp) if inp.suffix.lower()=='.xlsx' else pd.read_csv(inp)
    result=predict_dataframe(frame,args.strain,args.model);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    result.to_excel(out,index=False) if out.suffix.lower()=='.xlsx' else result.to_csv(out,index=False,encoding='utf-8-sig')
    print('Saved',len(result),'rows to',out)

if __name__=='__main__':main()
