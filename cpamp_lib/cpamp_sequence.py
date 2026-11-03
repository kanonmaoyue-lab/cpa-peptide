"""Versioned residue projection shared by repaired training and inference.

Lower-case D residues are projected onto parent residue identity, without claiming
stereochemistry is preserved by these 31 sequence descriptors. X remains unknown.
"""
from pathlib import Path
from functools import lru_cache
import importlib.util,re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
FEATURE_VERSION='sequence_resolved_v2'
AA='ACDEFGHIKLMNPQRSTVWY'

def normalize_sequence(value):
    if value is None or not isinstance(value,str):
        return ''
    s=value.strip()
    if s.lower() in ('','nan','none','null','na'):return ''
    if not re.fullmatch('[ACDEFGHIKLMNPQRSTVWYXacdefghiklmnpqrstvwyx]+',s):return ''
    return s.upper()

def resolve_qualitative(row):
    for name in ('tran_one_letter','one_letter','sequence'):
        seq=normalize_sequence(row.get(name))
        if seq:return seq,name
    return '','missing'

def database_sequence_lookup(raw=None):
    if raw is None:raw=pd.read_csv(ROOT/'datasets/DBAASP_full/dbaasp_cyclic_all_types.csv',dtype={'dbaasp_id':str})
    result={}
    for _,r in raw.iterrows():
        pid=str(r.dbaasp_id);seq=normalize_sequence(r.sequence)
        if pid in result and result[pid] and seq and result[pid]!=seq:
            raise ValueError(f'Sequence conflict for DBAASP id {pid}')
        if seq or pid not in result:result[pid]=seq
    return result

@lru_cache(maxsize=1)
def feature_module():
    spec=importlib.util.spec_from_file_location('cpamp_legacy_features',ROOT/'cpamp_lib/03_features.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def descriptor_block(value):
    """31 nominal residue descriptors; unknown chemistry stays explicitly missing."""
    seq=normalize_sequence(value)
    sf=feature_module().sequence_features(seq)
    names=['seq_len']+[f'aa_{a}' for a in AA]+[
        'seq_mw','seq_charge','seq_gravy','seq_aromatic_frac','seq_polar_frac',
        'seq_pos_frac','seq_neg_frac','seq_net_charge_ph7','seq_hydro_frac','seq_entropy']
    if 'X' in seq:
        # Composition uses total residue count; X mass/charge cannot be inferred.
        for key in names[21:30]:sf[key]=np.nan
    return np.asarray([sf[k] for k in names],dtype=np.float32)

def replace_sequence_block(X,sequences):
    if len(X)!=len(sequences):raise ValueError('Sequence rows must align with feature rows')
    out=np.asarray(X,dtype=np.float32).copy()
    out[:,:31]=np.vstack([descriptor_block(s) for s in sequences])
    if np.isinf(out).any():raise ValueError('Infinite features are not allowed')
    return out
