"""Reproducible, checkpointed sequence-repair experiments without overwriting v1."""
from pathlib import Path
import os,sys,json,hashlib,time,argparse,importlib.util,pickle
ROOT=Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT.parent/'.venv/Lib/site-packages'))
os.environ.setdefault('OMP_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,roc_auc_score,average_precision_score,matthews_corrcoef
from cpamp_sequence import normalize_sequence,resolve_qualitative,database_sequence_lookup,replace_sequence_block,FEATURE_VERSION
OUT=ROOT/'artifacts/sequence_repair_v2'
OUT.mkdir(exist_ok=True)
TASK_COLS=['anti_bacterial_any','anti_bacterial','anti_gram_pos','anti_gram_neg','anti_fungal']
TASKS=['Any antibacterial','Antibacterial','Gram-positive','Gram-negative','Antifungal']
THREADS=6

def load_script(filename):
    name='cpamp_source_'+filename.replace('.','_')
    spec=importlib.util.spec_from_file_location(name,ROOT/'cpamp_lib'/filename)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def log(*msg):print(time.strftime('%H:%M:%S'),*msg,flush=True)

def scaffold_folds(scafs,seed=7,nfold=5):
    values=np.asarray(scafs).astype(str)
    if np.any(values==''):raise ValueError('Empty scaffolds require explicit identity groups')
    vc=pd.Series(values).value_counts()
    rng=np.random.RandomState(seed)
    mapping={s:int(rng.randint(nfold)) for s in vc.index}
    return np.array([mapping[s] for s in values],dtype=np.int32)

def json_write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')

def corrected_delta(base,smiles,sequences,names):
    """Only update sequence-dependent delta fields; structural features stay frozen."""
    m30=load_script('30_delta_cyclization_features.py')
    sensitive=['dl_constraint_per_res','dl_ring_span_per_res','dl_flexibility_index',
               'dl_rigidity_per_res','dl_abs_charge_per_res',
               'dl_hydro_moment','dl_hydro_mean','dl_helix_propensity']
    identity_dependent={'dl_hydro_moment','dl_hydro_mean','dl_helix_propensity'}
    out=base.copy();positions={str(n):i for i,n in enumerate(names)}
    changed=[n for n in sensitive if n in positions]
    for i,(s,q) in enumerate(zip(smiles,sequences)):
        record=m30.delta_features(str(s),q)
        for name in changed:
            usable=bool(q) and not ('X' in q and name in identity_dependent)
            out[i,positions[name]]=record.get(name,np.nan) if usable else np.nan
        if (i+1)%1000==0:log('Sequence delta',i+1,'/',len(smiles))
    return out,changed

def prepare():
    from rdkit import Chem,RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
    ddir=ROOT/'datasets/DBAASP_full'
    lookup=database_sequence_lookup()
    z=np.load(ddir/'expanded.npz',allow_pickle=True)
    meta=pd.read_pickle(ddir/'expanded_meta.pkl').copy()
    ids=z['ids'].astype(str);byid=meta.drop_duplicates('dbaasp_id').assign(dbaasp_id=lambda d:d.dbaasp_id.astype(str)).set_index('dbaasp_id')
    seqs=[lookup.get(pid,'') for pid in ids]
    smiles=[byid.loc[pid,'smiles'] for pid in ids]
    old=z['X'].astype(np.float32)
    assert np.all(old[:,:31]==0)
    fixed=replace_sequence_block(old,seqs)
    delta,changed=corrected_delta(fixed,smiles,seqs,z['names'])
    log('Prepared quantitative features',fixed.shape,'sequence coverage',sum(bool(s) for s in seqs))
    scafs=meta._scaf.astype(str).to_numpy()
    if np.any(scafs==''):
        scafs=np.array([s if s else 'EMPTY_'+str(pid) for s,pid in zip(scafs,meta.dbaasp_id)])
    sf=scaffold_folds(scafs)
    np.savez_compressed(OUT/'quant_train.npz',X_legacy=old,X_fixed=fixed,X_delta=delta,ids=ids,
       sequences=np.array(seqs),smiles=np.array(smiles),xi=meta._xi.to_numpy(dtype=int),
       y=z['y'],strain=np.asarray(z['strain']).astype(str),bond=np.asarray(z['bond']).astype(str),
       fold=np.asarray(z['fold']),scaf=scafs,scaffold_fold=sf,names=z['names'])
    meta.to_csv(OUT/'quant_metadata.csv',index=False,encoding='utf-8-sig')
    m69=load_script('69_pantarget_qualitative.py')
    log('Resolving qualitative sequences and molecular features')
    qids,Q,Y,sub=m69.load_data()
    qseq=[resolve_qualitative(r)[0] for _,r in sub.iterrows()]
    qsource=[resolve_qualitative(r)[1] for _,r in sub.iterrows()]
    Qold=Q.copy();Qold[:,:31]=0
    qsc=[]
    for pid,smi in zip(qids,sub.smiles):
        mol=Chem.MolFromSmiles(smi)
        sc=MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else ''
        qsc.append(sc or 'EMPTY_'+pid)
    m98=load_script('98_qual_locto.py')
    bonds=np.array([m98.cyclo_type(s) for s in sub.smiles])
    yq=sub[TASK_COLS].astype(int).to_numpy()
    np.savez_compressed(OUT/'qual_train.npz',X_legacy=Qold,X_fixed=Q,ids=np.asarray(qids),
        sequences=np.array(qseq),smiles=sub.smiles.to_numpy(dtype=str),Y=yq,
        fold=sub.fold_cluster.to_numpy(),scaf=np.array(qsc),scaffold_fold=scaffold_folds(qsc),bond=bonds)
    qmeta=sub.reset_index()[['CPID','smiles','source','fold_cluster']+TASK_COLS].copy()
    qmeta['resolved_sequence']=qseq;qmeta['sequence_source']=qsource
    qmeta.to_csv(OUT/'qual_metadata.csv',index=False,encoding='utf-8-sig')
    # External labels are cached for the final locked evaluation, never read by development().
    e=pd.read_csv(ROOT/'artifacts/expanded/external_preds.csv',dtype={'dbaasp_id':str})
    raw=pd.read_csv(ddir/'mic_pairs.csv',dtype={'dbaasp_id':str})
    smiles_map=raw.drop_duplicates('dbaasp_id').set_index('dbaasp_id').smiles.to_dict()
    fmap={pid:old[i] for i,pid in enumerate(ids)}
    newids=[pid for pid in e.dbaasp_id.unique() if pid not in fmap]
    m90=load_script('90_build_expanded.py')
    if newids:
        Xe,keep=m90.compute_features([smiles_map[p] for p in newids],['']*len(newids))
        assert len(keep)==len(newids),'External feature loss must be investigated'
        for j,i in enumerate(keep):fmap[newids[i]]=Xe[j]
    Eold=np.vstack([fmap[p] for p in e.dbaasp_id])
    eseq=[lookup.get(pid,'') for pid in e.dbaasp_id]
    Efix=replace_sequence_block(Eold,eseq)
    Edelta,_=corrected_delta(Efix,[smiles_map[p] for p in e.dbaasp_id],eseq,z['names'])
    np.savez_compressed(OUT/'quant_external.npz',X_legacy=Eold,X_fixed=Efix,X_delta=Edelta,
        ids=e.dbaasp_id.to_numpy(dtype=str),y=e.log10_um.to_numpy(),strain=e.species.to_numpy(dtype=str),
        is_new=e.is_new_pep.to_numpy(),sequences=np.array(eseq))
    groups=raw.drop_duplicates('species').set_index('species').species_group.astype(str).to_dict()
    json_write(OUT/'strain_groups.json',groups)
    coverage=[]
    for channel,base,new,ss in [('quant',old,fixed,seqs),('qual',Qold,Q,qseq)]:
        for j in range(31):
            v=new[:,j];valid=np.isfinite(v)
            coverage.append({'channel':channel,'column':j,'rows':len(v),'n_valid':int(valid.sum()),
                'n_unique_finite':int(len(np.unique(v[valid]))),'all_zero_before':bool(np.all(base[:,j]==0)),
                'all_zero_after':bool(valid.all() and np.all(v==0))})
    pd.DataFrame(coverage).to_csv(OUT/'sequence_feature_audit.csv',index=False)
    assert not pd.DataFrame(coverage).all_zero_after.any()
    provenance={'feature_version':FEATURE_VERSION,'quant_shape':list(fixed.shape),'qual_shape':list(Q.shape),
      'quant_sequence_present':sum(bool(s) for s in seqs),'qual_sequence_present':sum(bool(s) for s in qseq),
      'quant_sequence_with_X':sum('X' in s for s in seqs),'qual_sequence_sources':pd.Series(qsource).value_counts().to_dict(),
      'qual_mature_projection_changes':sum(q!=normalize_sequence(r.get('sequence')) for q,(_,r) in zip(qseq,sub.iterrows())),
      'delta_updated_columns':changed,'external_rows':len(e),
      'selection_scope':'scaffold_fold != 0 only; scaffold_fold == 0 is untouched lockbox',
      'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path(__file__),ROOT/'cpamp_lib/cpamp_sequence.py',ddir/'expanded.npz',ROOT/'datasets/dataset_mtl.pkl']}}
    json_write(OUT/'data_manifest.json',provenance);log('PREPARE COMPLETE',provenance)

def ctx(train_strain,train_y,test_strain,groups):
    """All target-derived statistics are fitted on training observations only."""
    order=['gram_pos','gram_neg','fungal','other']
    codes={s:i+1 for i,s in enumerate(sorted(set(train_strain)))}
    median={s:float(np.median(train_y[train_strain==s])) if np.sum(train_strain==s)>=3 else 0. for s in codes}
    def transform(sp):
        A=np.zeros((len(sp),6),np.float32)
        for i,s in enumerate(sp):
            g=groups.get(str(s),'other');A[i,order.index(g) if g in order else 3]=1
            A[i,4]=codes.get(str(s),0)
            A[i,5]=median.get(str(s),0)
        return A
    return transform(train_strain),transform(test_strain)

def reg_metrics(y,p):
    return {'R2':float(r2_score(y,p)),'MAE':float(mean_absolute_error(y,p)),
            'within_1log':float(np.mean(np.abs(y-p)<=1)),'n':len(y)}

def cls_metrics(Y,P):
    rows=[]
    for j,t in enumerate(TASKS):
        yt=Y[:,j];pv=P[:,j]
        rows.append({'task':t,'AUROC':float(roc_auc_score(yt,pv)) if len(np.unique(yt))>1 else np.nan,
          'AUPRC':float(average_precision_score(yt,pv)) if yt.sum() else np.nan,
          'MCC':float(matthews_corrcoef(yt,pv>=.5)),'n':len(yt),'n_pos':int(yt.sum())})
    return rows

def fit_reg(X,y,sp,tr,te,groups,kind,seed=0):
    import lightgbm as lgb
    reg=kind=='regularized'
    params={'n_estimators':600,'learning_rate':.05,'num_leaves':31 if reg else 63,
      'min_child_samples':30 if reg else 15,'reg_lambda':2.0 if reg else 0.,
      'random_state':seed,'verbosity':-1,'colsample_bytree':.7,'subsample':.8,'subsample_freq':1,
      'n_jobs':THREADS,'force_col_wise':True}
    a,b=ctx(sp[tr],y[tr],sp[te],groups)
    mdl=lgb.LGBMRegressor(**params)
    mdl.fit(np.hstack([X[tr],a]),y[tr],categorical_feature=[X.shape[1]+4])
    return mdl.predict(np.hstack([X[te],b])),mdl

def long(X,Y):
    return np.hstack([np.repeat(X,5,axis=0),np.tile(np.eye(5,dtype=np.float32),(len(X),1))]),Y.ravel()

def fit_cls(X,Y,tr,te,kind,seed=0):
    import lightgbm as lgb
    reg=kind=='regularized'
    A,lab=long(X[tr],Y[tr]);B,_=long(X[te],Y[te])
    mdl=lgb.LGBMClassifier(n_estimators=500,learning_rate=.05,num_leaves=15 if reg else 31,
        min_child_samples=30 if reg else 20,reg_lambda=2.0 if reg else 0.,
        colsample_bytree=.8 if reg else 1.,random_state=seed,n_jobs=THREADS,verbosity=-1,force_col_wise=True)
    mdl.fit(A,lab)
    return mdl.predict_proba(B)[:,1].reshape(-1,5),mdl

def channel_data(channel):
    z=np.load(OUT/('quant_train.npz' if channel=='quant' else 'qual_train.npz'),allow_pickle=True)
    return {k:z[k] for k in z.files}

def candidate_X(d,candidate):
    key='X_legacy' if candidate=='legacy' else 'X_delta' if candidate=='fixed_delta' else 'X_fixed'
    X=d[key]
    return X[d['xi']] if 'xi' in d else X

def development(channel):
    """Fixed candidate screen within an 80% scaffold development pool only."""
    d=channel_data(channel);groups=json.loads((OUT/'strain_groups.json').read_text(encoding='utf8'))
    pool=d['scaffold_fold']!=0
    inner=scaffold_folds(d['scaf'],seed=171,nfold=3)
    candidates=['legacy','fixed','regularized']+(['fixed_delta'] if channel=='quant' else [])
    rows=[]
    for candidate in candidates:
        X=candidate_X(d,candidate)
        for f in range(3):
            tr=pool&(inner!=f);te=pool&(inner==f)
            p=OUT/f'dev_{channel}_{candidate}_{f}.npz'
            if p.exists():
                P=np.load(p)['prediction'];log('RESUME',p.name)
            else:
                log('DEV',channel,candidate,f,'train',int(tr.sum()),'test',int(te.sum()))
                P,_=fit_reg(X,d['y'],d['strain'],tr,te,groups,candidate) if channel=='quant' else fit_cls(X,d['Y'],tr,te,candidate)
                np.savez_compressed(p,prediction=P,test_index=np.flatnonzero(te))
            if channel=='quant':rows.append({'candidate':candidate,'fold':f,**reg_metrics(d['y'][te],P)})
            else:rows.extend([{'candidate':candidate,'fold':f,**m} for m in cls_metrics(d['Y'][te],P)])
        pd.DataFrame(rows).to_csv(OUT/f'development_{channel}.csv',index=False)
    frame=pd.DataFrame(rows)
    if channel=='quant':
        summary=frame.groupby('candidate')[['R2','MAE']].mean()
        eligible=summary[(summary.R2>=summary.loc['legacy','R2']+.02)&(summary.MAE<=summary.loc['legacy','MAE'])]
        chosen=eligible.R2.idxmax() if len(eligible) else 'legacy'
    else:
        summary=frame.groupby('candidate')[['AUROC','MCC']].mean()
        eligible=summary[summary.AUROC>=summary.loc['legacy','AUROC']+.005]
        chosen=eligible.AUROC.idxmax() if len(eligible) else 'legacy'
    result={'channel':channel,'selected_candidate':chosen,'development_summary':summary.reset_index().to_dict(orient='records'),
       'frozen':True,'selection_used_external_labels':False,'lockbox_used_for_selection':False,
       'note':'Selected solely within scaffold-folds 1-4. All final comparison protocols still required.'}
    json_write(OUT/f'selection_{channel}.json',result)
    log('DEVELOPMENT COMPLETE',channel,result)

def protocol_masks(d,protocol):
    n=len(d['fold'])
    if protocol=='lockbox':
        yield 'scaffold0',d['scaffold_fold']!=0,d['scaffold_fold']==0
    elif protocol in ('peptide_cv','scaffold_cv'):
        fl=d['fold'] if protocol=='peptide_cv' else d['scaffold_fold']
        for f in sorted(set(fl)):
            yield str(f),fl!=f,fl==f
    elif protocol=='locto':
        counts=pd.Series(d['bond']).value_counts()
        for value in counts[counts>=50].index:
            yield str(value),d['bond']!=value,d['bond']==value
    elif protocol=='family_holdout':
        counts=pd.Series(d['scaf']).value_counts()
        for j,value in enumerate(counts[counts>=10].index):
            yield str(j),d['scaf']!=value,d['scaf']==value
    else:raise ValueError(protocol)

def evaluate_one(channel,candidate,protocol):
    d=channel_data(channel);X=candidate_X(d,candidate)
    groups=json.loads((OUT/'strain_groups.json').read_text(encoding='utf8'))
    n=len(d['fold']);shape=(n,3) if channel=='quant' else (n,5,3)
    allpred=np.full(shape,np.nan,dtype=float)
    checkpoint=OUT/'checkpoints';checkpoint.mkdir(exist_ok=True)
    folds=[]
    for tag,tr,te in protocol_masks(d,protocol):
        if protocol in ('scaffold_cv','lockbox','family_holdout'):
            assert not set(d['scaf'][tr]).intersection(set(d['scaf'][te]))
        if protocol=='peptide_cv':
            pairids=d['ids'][d['xi']] if channel=='quant' else d['ids']
            assert not set(pairids[tr]).intersection(set(pairids[te]))
        folds.append({'fold':tag,'n_train':int(tr.sum()),'n_test':int(te.sum()),
                      'scaffold_overlap':len(set(d['scaf'][tr]).intersection(set(d['scaf'][te])))})
        for seed in range(3):
            p=checkpoint/f'{channel}_{candidate}_{protocol}_{tag}_{seed}.npz'
            if p.exists():
                cache=np.load(p);assert np.array_equal(cache['test_index'],np.flatnonzero(te))
                pred=cache['prediction'];log('RESUME',p.name)
            else:
                log('EVALUATE',channel,candidate,protocol,tag,'seed',seed)
                if channel=='quant':pred,_=fit_reg(X,d['y'],d['strain'],tr,te,groups,candidate,seed)
                else:pred,_=fit_cls(X,d['Y'],tr,te,candidate,seed)
                assert np.isfinite(pred).all()
                np.savez_compressed(p,prediction=pred,test_index=np.flatnonzero(te))
            if channel=='quant':allpred[te,seed]=pred
            else:allpred[te,:,seed]=pred
    valid=np.isfinite(allpred).all(axis=1) if channel=='quant' else np.isfinite(allpred).all(axis=(1,2))
    rows=[]
    for seed in range(3):
        if channel=='quant':
            rows.append({'channel':channel,'candidate':candidate,'protocol':protocol,'seed':seed,
                         **reg_metrics(d['y'][valid],allpred[valid,seed])})
        else:
            rows.extend([{'channel':channel,'candidate':candidate,'protocol':protocol,'seed':seed,**m}
                         for m in cls_metrics(d['Y'][valid],allpred[valid,:,seed])])
    meanpred=np.mean(allpred[valid],axis=-1)
    if channel=='quant':
        rows.append({'channel':channel,'candidate':candidate,'protocol':protocol,'seed':'ensemble',
                     **reg_metrics(d['y'][valid],meanpred)})
        pairids=d['ids'][d['xi']]
        source=pd.DataFrame({'dbaasp_id':pairids[valid],'strain':d['strain'][valid],
            'y':d['y'][valid],'prediction':meanpred,'bond':d['bond'][valid],'scaffold':d['scaf'][valid],
            **{f'pred_seed_{j}':allpred[valid,j] for j in range(3)}})
    else:
        rows.extend([{'channel':channel,'candidate':candidate,'protocol':protocol,'seed':'ensemble',**m}
                     for m in cls_metrics(d['Y'][valid],meanpred)])
        source=pd.concat([pd.DataFrame({'CPID':d['ids'][valid],'task':t,'y':d['Y'][valid,j],
             'prediction':meanpred[:,j],'bond':d['bond'][valid],'scaffold':d['scaf'][valid],
             **{f'pred_seed_{s}':allpred[valid,j,s] for s in range(3)}}) for j,t in enumerate(TASKS)],ignore_index=True)
    pd.DataFrame(rows).to_csv(OUT/f'eval_{channel}_{candidate}_{protocol}.csv',index=False)
    source.to_csv(OUT/f'oof_{channel}_{candidate}_{protocol}.csv',index=False,encoding='utf-8-sig')
    if protocol=='family_holdout' and channel=='qual':
        family_rows=[]
        for (scaf,task),sub in source.groupby(['scaffold','task'],sort=False):
            if sub.y.nunique()<2:continue
            family_rows.append({'candidate':candidate,'scaffold':scaf,'task':task,'n':len(sub),
                'AUROC':float(roc_auc_score(sub.y,sub.prediction)),
                'MCC':float(matthews_corrcoef(sub.y,sub.prediction>=.5))})
        pd.DataFrame(family_rows).to_csv(OUT/f'family_metrics_{candidate}.csv',index=False)
    json_write(OUT/f'folds_{channel}_{candidate}_{protocol}.json',
        {'folds':folds,'input_rows':n,'evaluated_rows':int(valid.sum()),'excluded_rows':int((~valid).sum()),
         'reason':'lockbox development rows not evaluated' if protocol=='lockbox' else 'types with fewer than 50 observations' if protocol=='locto' else 'families with fewer than 10 peptides' if protocol=='family_holdout' else 'none'})
    log('PROTOCOL COMPLETE',channel,candidate,protocol,[r for r in rows if r['seed']=='ensemble'])

def fit_final(channel,candidate):
    d=channel_data(channel);X=candidate_X(d,candidate)
    groups=json.loads((OUT/'strain_groups.json').read_text(encoding='utf8'))
    models=[]
    for seed in range(3):
        log('FINAL FIT',channel,candidate,seed)
        full=np.ones(len(d['fold']),bool)
        if channel=='quant':_,model=fit_reg(X,d['y'],d['strain'],full,full,groups,candidate,seed)
        else:_,model=fit_cls(X,d['Y'],full,full,candidate,seed)
        models.append(model)
    obj={'models':models,'feature_version':FEATURE_VERSION if candidate!='legacy' else 'legacy_zero_sequence',
      'candidate':candidate,'channel':channel,'n_feature':X.shape[1],'task_cols':TASK_COLS,'tasks':TASKS,
      'sequence_dependent_delta':candidate=='fixed_delta','training_ids':d['ids'],
      'training_smiles':d['smiles'],'training_sequences':d['sequences'],'schema_version':2}
    if channel=='quant':
        codes={s:i+1 for i,s in enumerate(sorted(set(d['strain'])))}
        obj.update({'strain_codes':codes,'strain_groups':groups,
                    'strain_median':{s:float(np.median(d['y'][d['strain']==s])) for s in codes}})
    path=OUT/f'model_{channel}_{candidate}.pkl'
    path.write_bytes(pickle.dumps(obj,protocol=pickle.HIGHEST_PROTOCOL))
    if channel=='quant':
        ext=np.load(OUT/'quant_external.npz',allow_pickle=True)
        key='X_legacy' if candidate=='legacy' else 'X_delta' if candidate=='fixed_delta' else 'X_fixed'
        E=ext[key];_,C=ctx(d['strain'],d['y'],ext['strain'],groups)
        P=np.column_stack([m.predict(np.hstack([E,C])) for m in models])
        pred=P.mean(axis=1)
        frame=pd.DataFrame({'dbaasp_id':ext['ids'],'strain':ext['strain'],'y':ext['y'],
            'is_new_peptide':ext['is_new'],'prediction':pred,**{f'pred_seed_{j}':P[:,j] for j in range(3)}})
        frame.to_csv(OUT/f'external_{candidate}.csv',index=False)
        rows=[]
        for subset,mask in [('all',np.ones(len(pred),bool)),('unseen_peptides',ext['is_new']),('known_peptides',~ext['is_new'])]:
            rows.append({'candidate':candidate,'subset':subset,**reg_metrics(ext['y'][mask],pred[mask])})
        for value in sorted(set(ext['strain'])):
            mask=ext['strain']==value
            rows.append({'candidate':candidate,'subset':value,**reg_metrics(ext['y'][mask],pred[mask])})
        pd.DataFrame(rows).to_csv(OUT/f'external_metrics_{candidate}.csv',index=False)
        log('EXTERNAL COMPLETE',candidate,rows[:3])
    log('FINAL MODEL SAVED',str(path))

def evaluate(channel):
    selection=json.loads((OUT/f'selection_{channel}.json').read_text(encoding='utf8'))
    chosen=selection['selected_candidate']
    candidates=list(dict.fromkeys(['legacy','fixed',chosen]))
    for candidate in candidates:
        for protocol in ['lockbox','peptide_cv','scaffold_cv','locto']:
            evaluate_one(channel,candidate,protocol)
        fit_final(channel,candidate)
    log('CHANNEL EVALUATION COMPLETE',channel)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','develop','evaluate','family']);ap.add_argument('--channel',choices=['qual','quant'])
    args=ap.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='family':
        for candidate in ['legacy','fixed']:evaluate_one('qual',candidate,'family_holdout')
    elif args.action in ('develop','evaluate'):
        if not args.channel:ap.error('--channel required')
        development(args.channel) if args.action=='develop' else evaluate(args.channel)

if __name__=='__main__':main()
