#

import holoclean
from detect import *
from repair.featurize import *

# 1. Setup a HoloClean session.
hc = holoclean.HoloClean(
    db_name='holo',
    domain_thresh_1=0.0,
    domain_thresh_2=0.0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.6,
    nb_cor_strength=0.8,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=1,
    batch_size=1,
    verbose=False,
    timeout=3 * 60000,
    print_fw=True,
    estimator_type='TupleEmbedding',
    train_attrs='age,workclass,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income'.split(','),
    infer_mode='dk'

).session

dset_name = 'adult_nonulls_sample10'

# 2. Load training data and denial constraints.
hc.load_data(f'{dset_name}', f'../testdata/{dset_name}/{dset_name}_all_columns_20.csv')
hc.load_dcs(f'../testdata/{dset_name}/{dset_name}_constraints.txt')
hc.ds.set_constraints(hc.get_dcs())

# 3. Detect erroneous cells using these two detectors.
detectors = [NullDetector(), ViolationDetector()]
# detectors = [NullDetector()]
hc.detect_errors(detectors)

# 4. Repair errors utilizing the defined features.
hc.generate_domain()
hc.run_estimator()
featurizers = [
    OccurAttrFeaturizer(),
    FreqFeaturizer(),
    ConstraintFeaturizer(),
]

hc.repair_errors(featurizers)

# 5. Evaluate the correctness of the results.
report = hc.evaluate(fpath=f'../testdata/{dset_name}/{dset_name}_ground-truth.csv',
            tid_col='tid',
            attr_col='attribute',
            val_col='correct_val')

df_pred = hc.get_predictions()
#
df_pred.to_csv(f'../testdata/{dset_name}/{dset_name}_pred.csv', index=False)