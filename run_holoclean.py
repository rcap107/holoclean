import argparse
import os
import sys
# sys.path.append('../../holoclean')

import pandas as pd
import pickle

import holoclean
from detect import *
from repair.featurize import *
from meta_data import ds_mdata, ds_natural

"""
Instructions: modify experiment parameters as necessary below. These will
be included in the appropriate dump filepaths to "cache" results.

Then run this script:
    python run_holoclean.py <exp-name> <ds-name> <attr-idx>
"""

### Experiment parameters (modify these per each run)

# These parameters are passed in via argparse (suggested defaults below)
# bin_size = 100
# embed_sz = 64
# max_domain = 50
# wdecay = 0.

# seed = 45
# cor_strength = 0.
hc_epochs = 20
embed_lr = 0.05
embed_epochs = 20
embed_batch_sz = 32
embed_lambda = 0.
infer_mode='all'
weak_label = False
wl_thresh = 0.9
estimator_type = 'Logistic'
init_feat = False

# Additional tokens to append to dump_prefix and preds_fpath.
addn_dump_toks = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_domain', metavar='max-domain', type=int, default=50)
    parser.add_argument('--wdecay', metavar='wdecay', type=float, default=0.)
    parser.add_argument('--bin_size', metavar='bin-size', type=int, default=100)
    parser.add_argument('--embed_sz', metavar='embed-sz', type=int, default=64)
    parser.add_argument('--cor', metavar='cor_strength', type=float, default=0.)
    parser.add_argument('--error_seed', metavar='miss-seed', type=int, default=42)
    parser.add_argument('--seed', metavar='random-seed', type=int, default=45,
                        help='random seed of hc')
    parser.add_argument('--validate_epoch', metavar='validate-epoch', type=int, default=30)
    parser.add_argument('--embed_dropout', metavar='embed-dropout', type=float, default=0.)
    parser.add_argument('exp_type', metavar='exp-type', type=str,)
    parser.add_argument('ds_name', metavar='ds-name', type=str,)
    parser.add_argument('attr_idx', metavar='attr-idx', type=int,
        help='Target attribute to train from dataset list of train attrs in ds_mdata. -1 will train on all attrs.')
    parser.add_argument('db_name_suffix', metavar='db-name-suffix', type=str,
        help='Suffix to append to holo for the db name.')

    args = parser.parse_args()
    exp_type = args.exp_type
    ds_name = args.ds_name
    attr_idx = args.attr_idx
    db_name_suffix = args.db_name_suffix
    error_seed = args.error_seed if ds_name not in ds_natural else ''
    seed = args.seed
    validate_epoch = args.validate_epoch

# Hyperparameters from argparse
    max_domain = args.max_domain
    wdecay = args.wdecay
    bin_size = args.bin_size
    embed_sz = args.embed_sz
    cor_strength = args.cor
    embed_dropout = args.embed_dropout

    dsdata = ds_mdata[ds_name]
    target_attrs = dsdata['target_attrs']
    num_attrs = dsdata['num_attrs']
    num_attr_groups = dsdata['num_attr_groups']
    data_dir = dsdata['data_dir']
    raw_prefix = dsdata['raw_prefix']
    clean_prefix = dsdata['clean_prefix']
    dc_file = dsdata['dc_file']
    hc_batch = dsdata['hc_batch']
    quantize_list = [(bin_size, [attr]) for attr in num_attrs]

    assert exp_type in ['hc', 'hcq', 'hce']
    assert attr_idx in range(-1, len(target_attrs))
    assert embed_dropout >= 0. and embed_dropout < 1



# Train model on the given attribute
    if attr_idx == -1:
        train_attrs = target_attrs.copy()
        attrs_tok = ['all']
    else:
        train_attrs = [target_attrs[attr_idx]]
        attrs_tok = train_attrs.copy()

### Set up filepaths to dump to.

    dom_fpath_toks = [ds_name, 'error', error_seed, 'seed', seed, 'cor', cor_strength, 'mode', infer_mode, 'maxdom', max_domain]
    cor_fpath_toks = [ds_name, 'error', error_seed]
    quant_fpath_toks = [ds_name, 'error', error_seed, 'seed', seed]
    dump_prefix_toks = dom_fpath_toks.copy() + addn_dump_toks + ['exp', exp_type]
    if exp_type in ['hcq', 'hce']:
        dom_fpath_toks += ['bin', bin_size]
        cor_fpath_toks += ['bin', bin_size]
        quant_fpath_toks += ['bin', bin_size]
        dump_prefix_toks += ['bin', bin_size]

        if weak_label:
            dom_fpath_toks += ['wl', wl_thresh, estimator_type]

        if exp_type == 'hce':
            dump_prefix_toks += ['elr', embed_lr, 'eepochs', embed_epochs, 'ebatch', embed_batch_sz, 'elambda', embed_lambda, 'esz', embed_sz]
            if embed_dropout > 0.:
                dump_prefix_toks += ['dropout', embed_dropout]
    preds_fpath_toks = dump_prefix_toks.copy() + ['wd', wdecay, 'hcepochs', hc_epochs, 'initfeat', init_feat]
    if dc_file is not None:
        preds_fpath_toks += ['dc', dc_file.split('.txt')[0]]

    dom_fpath_toks += ['attrs'] + attrs_tok
    dump_prefix_toks += ['attrs'] + attrs_tok
    preds_fpath_toks += ['attrs'] + attrs_tok

    errors_fpath_toks = [ds_name, 'error', error_seed]
    # MVI: no DC errors
    # if dc_file is not None:
    #     errors_fpath_toks += [dc_file.split('.txt')[0]]

    dump_dir = '/home/spoutnik23/phd/holoclean/dump-dir/%s' % ds_name
    assert os.path.isdir(dump_dir)

    errors_fpath = '%s/%s_errors.pkl' % (dump_dir, '_'.join(map(str, errors_fpath_toks)))
    dom_fpath = '%s/%s_domain.pkl' % (dump_dir, '_'.join(map(str, dom_fpath_toks)))
    cor_fpath = '%s/%s_corrs.pkl' % (dump_dir, '_'.join(map(str, cor_fpath_toks)))
    quant_fpath = '%s/%s_quantized_raw_df.pkl' % (dump_dir, '_'.join(map(str, quant_fpath_toks)))
    dump_prefix = '%s/%s' % (dump_dir, '_'.join(map(str, dump_prefix_toks)))
    preds_fpath = '%s/%s_%s_imputed_holoclean.csv' % (dump_dir, exp_type, raw_prefix)
    # preds_fpath = '%s/%s_preds.csv' % (dump_dir, '_'.join(map(str, preds_fpath_toks)))

    print('Exp type:', exp_type)
    print('Errors fpath:', errors_fpath)
    print('Dom fpath:', dom_fpath)
    print('Corrs fpath:', cor_fpath)
    print('Quantized fpath:', quant_fpath)
    print('Dump prefix (only hce):', dump_prefix)
    print('Preds fpath:', preds_fpath)

### Setup a HoloClean session.
    hc = holoclean.HoloClean(
        db_name='holo' + db_name_suffix,
        seed=seed,
        domain_thresh_1=0.0,
        domain_thresh_2=0.0,
        weak_label_thresh=wl_thresh,
        epochs=hc_epochs,
        max_domain=max_domain,
        cor_strength=cor_strength,
        weight_decay=wdecay,
        learning_rate=0.001,
        threads=1,
        batch_size=hc_batch,
        verbose=True,
        timeout=3 * 60000,
        print_fw=True,
        train_attrs=train_attrs,
        infer_mode=infer_mode,
        estimator_type=estimator_type,
        estimator_embedding_size=embed_sz,
    ).session

### Load training data and denial constraints
    hc.load_data(raw_prefix.replace('.', '_').replace(' ', '_')[:50],
                 '{data_dir}/{pre}{error_seed}.csv'.format(data_dir=data_dir, pre=raw_prefix, error_seed=error_seed),
                 numerical_attrs=num_attrs,
                 )

### Load DCs and detect DK cells

    detectors = [NullDetector()]
    if dc_file is not None:
        hc.load_dcs('{data_dir}/{dc_file}'.format(data_dir=data_dir, dc_file=dc_file))
        hc.ds.set_constraints(hc.get_dcs())
        # MVI: no DC errors
        # detectors += [ViolationDetector()]

    if os.path.exists(errors_fpath):
        hc.detect_engine.errors_df = pd.read_pickle(errors_fpath)
        hc.detect_engine.store_detected_errors(hc.detect_engine.errors_df)
    else:
        hc.detect_errors(detectors)
        hc.detect_engine.errors_df.to_pickle(errors_fpath)


### Quantization
    if exp_type in ['hcq', 'hce']:
        if os.path.exists(quant_fpath):
            quant_df = pd.read_pickle(quant_fpath)
            hc.load_quantized_data(quant_df)
        else:
            quant_df = hc.quantize_numericals(quantize_list)
            quant_df.to_pickle(quant_fpath)

### Correlations

    if os.path.exists(cor_fpath):
        hc.domain_engine.correlations = pickle.load(open(cor_fpath, 'rb'))
    else:
        hc.domain_engine.compute_correlations()
        pickle.dump(hc.domain_engine.correlations, open(cor_fpath, 'wb'))

### Domain generation

    if os.path.exists(dom_fpath):
        hc.domain_engine.domain_df = pd.read_pickle(dom_fpath)
        hc.domain_engine.store_domains(hc.domain_engine.domain_df)
    else:
        hc.generate_domain()
        if weak_label:
            hc.run_estimator()
        hc.domain_engine.domain_df.to_pickle(dom_fpath)


### Featurizers and repair

    featurizers = []
    if init_feat:
        featurizers += [InitAttrFeaturizer()]
    if dc_file is not None:
        featurizers += [ConstraintFeaturizer()]

# Use embedding model as 'co-occurrence' for 'hce': otherwise use Cooccurrence
    if exp_type == 'hce':
        # Quantize only for domain generation + stats
        hc.disable_quantize()

        featurizers.insert(0, EmbeddingFeaturizer(dump_prefix=dump_prefix,
            dropout_pct=embed_dropout,
            learning_rate=embed_lr,
            epochs=embed_epochs,
            batch_size=embed_batch_sz,
            weight_lambda=embed_lambda,
            numerical_attr_groups=num_attr_groups,
            # validate_fpath='{data_dir}/{pre}_clean.csv'.format(data_dir=data_dir, pre=clean_prefix),
            # validate_epoch=validate_epoch
        ))
    else:
        featurizers.insert(0, OccurAttrFeaturizer())
    print('repair errors')
    hc.repair_errors(featurizers)

### Dump predictions
    print('dump predictions')
    df_preds = hc.get_predictions()
    df_preds.to_csv(preds_fpath, index=False)

### Evaluate the correctness of the results.

    report = hc.evaluate(fpath='{data_dir}/{pre}_clean.csv'.format(data_dir=data_dir, pre=clean_prefix),
                tid_col='tid',
                attr_col='attribute',
                val_col='correct_val') 
