# meta_data.py

import pandas as pd

### Metdata per each dataset

__all__ = ['ds_natural', 'ds_mdata', 'get_errors_fpath',
           'build_hc_prefix', 'build_xgb_prefix', 'build_cgb_prefix']

# cor_strength = 0.
hc_epochs = 20
embed_lr = 0.05
embed_epochs = 20
embed_batch_sz = 32
embed_lambda = 0.
infer_mode = 'all'
weak_label = False
wl_thresh = 0.9
estimator_type = 'Logistic'
init_feat = False

dump_dir = '/home/hce-exp-files/'
# for debug
# dump_dir = ''

# all the ds with natural errors but not injected,
# they won't need error_seed
ds_natural = ['chicago_400k', 'chicago_10k']

ds_mdata = {
    'adult_nonulls_sample10': {
        'target_attrs': 'age,workclass,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income'.split(','),
        # 'target_attrs': ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation',
        #                  'relationship', 'race', 'sex',
        #                  'hours-per-week', 'native-country', 'income'],
        'num_attrs': ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
        'num_attr_groups': [['age'], ['education-num'], ['capital-gain'], ['capital-loss'], ['hours-per-week']],
        'data_dir': 'testdata/raw/adult_nonulls_sample10',
        'raw_prefix': 'adult_nonulls_sample10_all_columns_20',
        'clean_prefix': 'adult_nonulls_sample10',
        # 'dc_file': 'adult_constraints.txt',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },

    'contraceptive': {
        'target_attrs': 'A1,A2,A3,A4,A5,A6,A7,A8,A9,A10'.split(','),
        'num_attrs': ['A1','A4'],
        'num_attr_groups': [['A1'], ['A4']],
        'data_dir': 'testdata/raw/contraceptive',
        'raw_prefix': 'contraceptive_all_columns_60',
        'clean_prefix': 'contraceptive',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },
    'imdb': {
        'target_attrs': 'color,director,actor_2,actor_1,title,actor_3,original_language,production_countries,content_rating,year,vote_average'.split(','),
        'num_attrs': ['year'],
        'num_attr_groups': [['year']],
        'data_dir': 'testdata/raw/imdb',
        'raw_prefix': 'imdb_all_columns_20',
        'clean_prefix': 'imdb',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },

    'tictactoe_20': {
        'target_attrs': 'A1,A2,A3,A4,A5,A6,A7,A8,A9'.split(','),
        'num_attrs': [],
        'num_attr_groups': [],
        'data_dir': 'testdata/raw/tictactoe',
        'raw_prefix': 'tictactoe_all_columns_20',
        'clean_prefix': 'tictactoe',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },

    'tictactoe_60': {
        'target_attrs': 'A1,A2,A3,A4,A5,A6,A7,A8,A9'.split(','),
        'num_attrs': [],
        'num_attr_groups': [],
        'data_dir': 'testdata/raw/tictactoe',
        'raw_prefix': 'tictactoe_all_columns_60',
        'clean_prefix': 'tictactoe',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },

    'australian_20': {
        'target_attrs': 'A1,A2,A3,A4,A5,A6,A7,A8,A9'.split(','),
        'num_attrs': ['A2', 'A3', 'A7', 'A13', 'A14'],
        'num_attr_groups': [['A2'], ['A3'], ['A7'], ['A13'], ['A14']],
        'data_dir': 'testdata/raw/australian',
        'raw_prefix': 'australian_all_columns_20',
        'clean_prefix': 'australian',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },

    'australian_60': {
        'target_attrs': 'A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15'.split(','),
        'num_attrs': [],
        'num_attr_groups': [],
        'data_dir': 'testdata/raw/australian',
        'raw_prefix': 'australian_all_columns_60',
        'clean_prefix': 'australian',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },

    'flare_60': {
        'target_attrs': 'A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13'.split(','),
        'num_attrs': ['A11', 'A12', 'A13'],
        'num_attr_groups': [['A11'], ['A12'], ['A13']],
        'data_dir': 'testdata/raw/flare',
        'raw_prefix': 'flare_all_columns_60',
        'clean_prefix': 'flare',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },
    'flare_20': {
        'target_attrs': 'A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13'.split(','),
        'num_attrs': ['A11', 'A12', 'A13'],
        'num_attr_groups': [['A11'], ['A12'], ['A13']],
        'data_dir': 'testdata/raw/flare',
        'raw_prefix': 'flare_all_columns_20',
        'clean_prefix': 'flare',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },

    'fodors_zagats': {
        'target_attrs': 'city, restaurant_type'.split(','),
        'num_attrs': [],
        'num_attr_groups': [],
        'data_dir': 'testdata/raw/fodors_zagats',
        'raw_prefix': 'fodors_zagats_all_columns_20',
        'clean_prefix': 'fodors_zagats',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },
    'bikes_2': {
        'target_attrs': 'id,bike_name,city_posted,km_driven,color,fuel_type,price,model_year,owner_type'.split(','),
        'num_attrs': ['km_driven', 'price', 'model_year'],
        'num_attr_groups': [['km_driven'], ['price'], ['model_year']],
        'data_dir': 'testdata/raw/bikes',
        'raw_prefix': 'bikes_all_columns_2',
        'clean_prefix': 'bikes',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },
    'bikes_20': {
        'target_attrs': 'id,bike_name,city_posted,km_driven,color,fuel_type,price,model_year,owner_type'.split(','),
        'num_attrs': ['km_driven', 'price', 'model_year'],
        'num_attr_groups': [['km_driven'], ['price'], ['model_year']],
        'data_dir': 'testdata/raw/bikes',
        'raw_prefix': 'bikes_all_columns_2',
        'clean_prefix': 'bikes',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },
    'bikes_10': {
        'target_attrs': 'id,bike_name,city_posted,km_driven,color,fuel_type,price,model_year,owner_type'.split(','),
        'num_attrs': ['km_driven', 'price', 'model_year'],
        'num_attr_groups': [['km_driven'], ['price'], ['model_year']],
        'data_dir': 'testdata/raw/bikes',
        'raw_prefix': 'bikes_all_columns_2',
        'clean_prefix': 'bikes',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },
    'bikes_dekho_2': {
        'target_attrs': 'id,bike_name,city_posted,km_driven,color,fuel_type,price,model_year,owner_type'.split(','),
        'num_attrs': ['km_driven', 'price', 'model_year'],
        'num_attr_groups': [['km_driven'], ['price'], ['model_year']],
        'data_dir': 'testdata/raw/bikes_dekho',
        'raw_prefix': 'bikes_dekho_all_columns_2',
        'clean_prefix': 'bikes_dekho',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },
    'bikes_dekho_20': {
        'target_attrs': 'id,bike_name,city_posted,km_driven,color,fuel_type,price,model_year,owner_type'.split(','),
        'num_attrs': ['km_driven', 'price', 'model_year'],
        'num_attr_groups': [['km_driven'], ['price'], ['model_year']],
        'data_dir': 'testdata/raw/bikes_dekho',
        'raw_prefix': 'bikes_dekho_all_columns_20',
        'clean_prefix': 'bikes_dekho',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },
    'bikes_dekho_10': {
        'target_attrs': 'id,bike_name,city_posted,km_driven,color,fuel_type,price,model_year,owner_type'.split(','),
        'num_attrs': ['km_driven', 'price', 'model_year'],
        'num_attr_groups': [['km_driven'], ['price'], ['model_year']],
        'data_dir': 'testdata/raw/bikes_dekho',
        'raw_prefix': 'bikes_dekho_all_columns_10',
        'clean_prefix': 'bikes_dekho',
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,

    },

}


def get_errors_fpath(ds_name, error_seed):
    if ds_name == 'chicago_400k':
        return None

    errors_fpath_toks = [ds_name, 'error', error_seed]
    dc_file = ds_mdata[ds_name]['dc_file']

    if dc_file is not None:
        errors_fpath_toks += [dc_file.split('.txt')[0]]

    errors_fpath = '%s/%s_errors.pkl' % (dump_dir + ds_name, '_'.join(map(str, errors_fpath_toks)))
    return errors_fpath


def build_hc_prefix(dump_dir, exp_type, ds_name, error_seed, seed,
                    max_domain, wdecay, bin_size, embed_sz, embed_dropout, cor):
    """
    Returns the prefix for (embedding model, FULL HC model)
    """
    ds_data = ds_mdata[ds_name]
    dc_file = ds_data['dc_file']

    dump_fpath_toks = [ds_name, 'error', error_seed, 'seed', seed, 'cor', cor,
                       'mode', infer_mode, 'maxdom', max_domain, 'exp', exp_type]

    if exp_type in ['hcq', 'hce']:
        dump_fpath_toks += ['bin', bin_size]

        if exp_type == 'hce':
            dump_fpath_toks += ['elr', embed_lr, 'eepochs', embed_epochs, 'ebatch', embed_batch_sz, 'elambda',
                                 embed_lambda, 'esz', embed_sz]
            if embed_dropout > 0.:
                dump_fpath_toks += ['dropout', embed_dropout]
    preds_fpath_toks = dump_fpath_toks.copy() + ['wd', wdecay, 'hcepochs', hc_epochs, 'initfeat', init_feat]
    if dc_file is not None:
        preds_fpath_toks += ['dc', dc_file.split('.txt')[0]]

    dump_fpath_toks += ['attrs']
    preds_fpath_toks += ['attrs']

    dump_fpath = '%s/%s/%s' % (dump_dir, ds_name, '_'.join(map(str, dump_fpath_toks)))
    preds_fpath = '%s/%s/%s' % (dump_dir, ds_name, '_'.join(map(str, preds_fpath_toks)))

    return dump_fpath, preds_fpath


def build_xgb_prefix(dump_dir, ds_name, error_seed, seed, **xgb_params):
    preds_fpath_toks = [ds_name, 'exp', 'xgb', 'error', error_seed, 'seed', seed,
                        'tree', xgb_params['n_estimators'],
                        'lr', xgb_params['eta'],
                        'depth', xgb_params['max_depth'],
                        'mcw', xgb_params['min_child_weight'],
                        'gamma', xgb_params['gamma'],
                        ]

    preds_prefix = '%s/%s/%s' % (dump_dir, ds_name, '_'.join(map(str, preds_fpath_toks)))

    return preds_prefix


def build_cgb_prefix(dump_dir, ds_name, error_seed, seed, **cgb_params):
    preds_fpath_toks = [ds_name, 'exp', 'cgb', 'error', error_seed, 'seed', seed,
                        'tree', cgb_params['n_estimators'],
                        'lr', cgb_params['eta'],
                        'depth', cgb_params['max_depth']
                        ]

    preds_prefix = '%s/%s/%s' % (dump_dir, ds_name, '_'.join(map(str, preds_fpath_toks)))

    return preds_prefix
 
