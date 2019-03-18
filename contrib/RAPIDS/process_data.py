import numpy as np
import datetime
import dask_xgboost as dxgb_gpu
import dask
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.delayed import delayed
from dask.distributed import Client, wait
import xgboost as xgb
import cudf
from cudf.dataframe import DataFrame
from collections import OrderedDict
import gc
from glob import glob
import os
import argparse

def initialize_rmm_pool():
    from librmm_cffi import librmm_config as rmm_cfg

    rmm_cfg.use_pool_allocator = True
    #rmm_cfg.initial_pool_size = 2<<30 # set to 2GiB. Default is 1/2 total GPU memory
    import cudf
    return cudf._gdf.rmm_initialize()

def initialize_rmm_no_pool():
    from librmm_cffi import librmm_config as rmm_cfg
    
    rmm_cfg.use_pool_allocator = False
    import cudf
    return cudf._gdf.rmm_initialize()

def run_dask_task(func, **kwargs):
    task = func(**kwargs)
    return task

def process_quarter_gpu(client, col_names_path, acq_data_path, year=2000, quarter=1, perf_file=""):
    dask_client = client
    ml_arrays = run_dask_task(delayed(run_gpu_workflow),
                                          col_path=col_names_path,
                                          acq_path=acq_data_path,
                                          quarter=quarter,
                                          year=year,
                                          perf_file=perf_file)
    return dask_client.compute(ml_arrays,
                          optimize_graph=False,
                          fifo_timeout="0ms")

def null_workaround(df, **kwargs):
    for column, data_type in df.dtypes.items():
        if str(data_type) == "category":
            df[column] = df[column].astype('int32').fillna(-1)
        if str(data_type) in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
            df[column] = df[column].fillna(-1)
    return df

def run_gpu_workflow(col_path, acq_path, quarter=1, year=2000, perf_file="", **kwargs):
    names = gpu_load_names(col_path=col_path)
    acq_gdf = gpu_load_acquisition_csv(acquisition_path= acq_path + "/Acquisition_"
                                      + str(year) + "Q" + str(quarter) + ".txt")
    acq_gdf = acq_gdf.merge(names, how='left', on=['seller_name'])
    acq_gdf.drop_column('seller_name')
    acq_gdf['seller_name'] = acq_gdf['new']
    acq_gdf.drop_column('new')
    perf_df_tmp = gpu_load_performance_csv(perf_file)
    gdf = perf_df_tmp
    everdf = create_ever_features(gdf)
    delinq_merge = create_delinq_features(gdf)
    everdf = join_ever_delinq_features(everdf, delinq_merge)
    del(delinq_merge)
    joined_df = create_joined_df(gdf, everdf)
    testdf = create_12_mon_features(joined_df)
    joined_df = combine_joined_12_mon(joined_df, testdf)
    del(testdf)
    perf_df = final_performance_delinquency(gdf, joined_df)
    del(gdf, joined_df)
    final_gdf = join_perf_acq_gdfs(perf_df, acq_gdf)
    del(perf_df)
    del(acq_gdf)
    final_gdf = last_mile_cleaning(final_gdf)
    return final_gdf

def gpu_load_performance_csv(performance_path, **kwargs):
    """ Loads performance data

    Returns
    -------
    GPU DataFrame
    """
    
    cols = [
        "loan_id", "monthly_reporting_period", "servicer", "interest_rate", "current_actual_upb",
        "loan_age", "remaining_months_to_legal_maturity", "adj_remaining_months_to_maturity",
        "maturity_date", "msa", "current_loan_delinquency_status", "mod_flag", "zero_balance_code",
        "zero_balance_effective_date", "last_paid_installment_date", "foreclosed_after",
        "disposition_date", "foreclosure_costs", "prop_preservation_and_repair_costs",
        "asset_recovery_costs", "misc_holding_expenses", "holding_taxes", "net_sale_proceeds",
        "credit_enhancement_proceeds", "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
        "non_interest_bearing_upb", "principal_forgiveness_upb", "repurchase_make_whole_proceeds_flag",
        "foreclosure_principal_write_off_amount", "servicing_activity_indicator"
    ]
    
    dtypes = OrderedDict([
        ("loan_id", "int64"),
        ("monthly_reporting_period", "date"),
        ("servicer", "category"),
        ("interest_rate", "float64"),
        ("current_actual_upb", "float64"),
        ("loan_age", "float64"),
        ("remaining_months_to_legal_maturity", "float64"),
        ("adj_remaining_months_to_maturity", "float64"),
        ("maturity_date", "date"),
        ("msa", "float64"),
        ("current_loan_delinquency_status", "int32"),
        ("mod_flag", "category"),
        ("zero_balance_code", "category"),
        ("zero_balance_effective_date", "date"),
        ("last_paid_installment_date", "date"),
        ("foreclosed_after", "date"),
        ("disposition_date", "date"),
        ("foreclosure_costs", "float64"),
        ("prop_preservation_and_repair_costs", "float64"),
        ("asset_recovery_costs", "float64"),
        ("misc_holding_expenses", "float64"),
        ("holding_taxes", "float64"),
        ("net_sale_proceeds", "float64"),
        ("credit_enhancement_proceeds", "float64"),
        ("repurchase_make_whole_proceeds", "float64"),
        ("other_foreclosure_proceeds", "float64"),
        ("non_interest_bearing_upb", "float64"),
        ("principal_forgiveness_upb", "float64"),
        ("repurchase_make_whole_proceeds_flag", "category"),
        ("foreclosure_principal_write_off_amount", "float64"),
        ("servicing_activity_indicator", "category")
    ])

    print(performance_path)
    
    return cudf.read_csv(performance_path, names=cols, delimiter='|', dtype=list(dtypes.values()), skiprows=1)

def gpu_load_acquisition_csv(acquisition_path, **kwargs):
    """ Loads acquisition data

    Returns
    -------
    GPU DataFrame
    """
    
    cols = [
        'loan_id', 'orig_channel', 'seller_name', 'orig_interest_rate', 'orig_upb', 'orig_loan_term', 
        'orig_date', 'first_pay_date', 'orig_ltv', 'orig_cltv', 'num_borrowers', 'dti', 'borrower_credit_score', 
        'first_home_buyer', 'loan_purpose', 'property_type', 'num_units', 'occupancy_status', 'property_state',
        'zip', 'mortgage_insurance_percent', 'product_type', 'coborrow_credit_score', 'mortgage_insurance_type', 
        'relocation_mortgage_indicator'
    ]
    
    dtypes = OrderedDict([
        ("loan_id", "int64"),
        ("orig_channel", "category"),
        ("seller_name", "category"),
        ("orig_interest_rate", "float64"),
        ("orig_upb", "int64"),
        ("orig_loan_term", "int64"),
        ("orig_date", "date"),
        ("first_pay_date", "date"),
        ("orig_ltv", "float64"),
        ("orig_cltv", "float64"),
        ("num_borrowers", "float64"),
        ("dti", "float64"),
        ("borrower_credit_score", "float64"),
        ("first_home_buyer", "category"),
        ("loan_purpose", "category"),
        ("property_type", "category"),
        ("num_units", "int64"),
        ("occupancy_status", "category"),
        ("property_state", "category"),
        ("zip", "int64"),
        ("mortgage_insurance_percent", "float64"),
        ("product_type", "category"),
        ("coborrow_credit_score", "float64"),
        ("mortgage_insurance_type", "float64"),
        ("relocation_mortgage_indicator", "category")
    ])
    
    print(acquisition_path)
    
    return cudf.read_csv(acquisition_path, names=cols, delimiter='|', dtype=list(dtypes.values()), skiprows=1)

def gpu_load_names(col_path):
    """ Loads names used for renaming the banks
    
    Returns
    -------
    GPU DataFrame
    """

    cols = [
        'seller_name', 'new'
    ]
    
    dtypes = OrderedDict([
        ("seller_name", "category"),
        ("new", "category"),
    ])

    return cudf.read_csv(col_path, names=cols, delimiter='|', dtype=list(dtypes.values()), skiprows=1)

def create_ever_features(gdf, **kwargs):
    everdf = gdf[['loan_id', 'current_loan_delinquency_status']]
    everdf = everdf.groupby('loan_id', method='hash').max()
    del(gdf)
    everdf['ever_30'] = (everdf['max_current_loan_delinquency_status'] >= 1).astype('int8')
    everdf['ever_90'] = (everdf['max_current_loan_delinquency_status'] >= 3).astype('int8')
    everdf['ever_180'] = (everdf['max_current_loan_delinquency_status'] >= 6).astype('int8')
    everdf.drop_column('max_current_loan_delinquency_status')
    return everdf

def create_delinq_features(gdf, **kwargs):
    delinq_gdf = gdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status']]
    del(gdf)
    delinq_30 = delinq_gdf.query('current_loan_delinquency_status >= 1')[['loan_id', 'monthly_reporting_period']].groupby('loan_id', method='hash').min()
    delinq_30['delinquency_30'] = delinq_30['min_monthly_reporting_period']
    delinq_30.drop_column('min_monthly_reporting_period')
    delinq_90 = delinq_gdf.query('current_loan_delinquency_status >= 3')[['loan_id', 'monthly_reporting_period']].groupby('loan_id', method='hash').min()
    delinq_90['delinquency_90'] = delinq_90['min_monthly_reporting_period']
    delinq_90.drop_column('min_monthly_reporting_period')
    delinq_180 = delinq_gdf.query('current_loan_delinquency_status >= 6')[['loan_id', 'monthly_reporting_period']].groupby('loan_id', method='hash').min()
    delinq_180['delinquency_180'] = delinq_180['min_monthly_reporting_period']
    delinq_180.drop_column('min_monthly_reporting_period')
    del(delinq_gdf)
    delinq_merge = delinq_30.merge(delinq_90, how='left', on=['loan_id'], type='hash')
    delinq_merge['delinquency_90'] = delinq_merge['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    delinq_merge = delinq_merge.merge(delinq_180, how='left', on=['loan_id'], type='hash')
    delinq_merge['delinquency_180'] = delinq_merge['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    del(delinq_30)
    del(delinq_90)
    del(delinq_180)
    return delinq_merge

def join_ever_delinq_features(everdf_tmp, delinq_merge, **kwargs):
    everdf = everdf_tmp.merge(delinq_merge, on=['loan_id'], how='left', type='hash')
    del(everdf_tmp)
    del(delinq_merge)
    everdf['delinquency_30'] = everdf['delinquency_30'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    everdf['delinquency_90'] = everdf['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    everdf['delinquency_180'] = everdf['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    return everdf

def create_joined_df(gdf, everdf, **kwargs):
    test = gdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status', 'current_actual_upb']]
    del(gdf)
    test['timestamp'] = test['monthly_reporting_period']
    test.drop_column('monthly_reporting_period')
    test['timestamp_month'] = test['timestamp'].dt.month
    test['timestamp_year'] = test['timestamp'].dt.year
    test['delinquency_12'] = test['current_loan_delinquency_status']
    test.drop_column('current_loan_delinquency_status')
    test['upb_12'] = test['current_actual_upb']
    test.drop_column('current_actual_upb')
    test['upb_12'] = test['upb_12'].fillna(999999999)
    test['delinquency_12'] = test['delinquency_12'].fillna(-1)
    
    joined_df = test.merge(everdf, how='left', on=['loan_id'], type='hash')
    del(everdf)
    del(test)
    
    joined_df['ever_30'] = joined_df['ever_30'].fillna(-1)
    joined_df['ever_90'] = joined_df['ever_90'].fillna(-1)
    joined_df['ever_180'] = joined_df['ever_180'].fillna(-1)
    joined_df['delinquency_30'] = joined_df['delinquency_30'].fillna(-1)
    joined_df['delinquency_90'] = joined_df['delinquency_90'].fillna(-1)
    joined_df['delinquency_180'] = joined_df['delinquency_180'].fillna(-1)
    
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int32')
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int32')
    
    return joined_df

def create_12_mon_features(joined_df, **kwargs):
    testdfs = []
    n_months = 12
    for y in range(1, n_months + 1):
        tmpdf = joined_df[['loan_id', 'timestamp_year', 'timestamp_month', 'delinquency_12', 'upb_12']]
        tmpdf['josh_months'] = tmpdf['timestamp_year'] * 12 + tmpdf['timestamp_month']
        tmpdf['josh_mody_n'] = ((tmpdf['josh_months'].astype('float64') - 24000 - y) / 12).floor()
        tmpdf = tmpdf.groupby(['loan_id', 'josh_mody_n'], method='hash').agg({'delinquency_12': 'max','upb_12': 'min'})
        tmpdf['delinquency_12'] = (tmpdf['max_delinquency_12']>3).astype('int32')
        tmpdf['delinquency_12'] +=(tmpdf['min_upb_12']==0).astype('int32')
        tmpdf.drop_column('max_delinquency_12')
        tmpdf['upb_12'] = tmpdf['min_upb_12']
        tmpdf.drop_column('min_upb_12')
        tmpdf['timestamp_year'] = (((tmpdf['josh_mody_n'] * n_months) + 24000 + (y - 1)) / 12).floor().astype('int16')
        tmpdf['timestamp_month'] = np.int8(y)
        tmpdf.drop_column('josh_mody_n')
        testdfs.append(tmpdf)
        del(tmpdf)
    del(joined_df)

    return cudf.concat(testdfs)

def combine_joined_12_mon(joined_df, testdf, **kwargs):
    joined_df.drop_column('delinquency_12')
    joined_df.drop_column('upb_12')
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int16')
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int8')
    return joined_df.merge(testdf, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'], type='hash')

def final_performance_delinquency(gdf, joined_df, **kwargs):
    merged = null_workaround(gdf)
    joined_df = null_workaround(joined_df)
    merged['timestamp_month'] = merged['monthly_reporting_period'].dt.month
    merged['timestamp_month'] = merged['timestamp_month'].astype('int8')
    merged['timestamp_year'] = merged['monthly_reporting_period'].dt.year
    merged['timestamp_year'] = merged['timestamp_year'].astype('int16')
    merged = merged.merge(joined_df, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'], type='hash')
    merged.drop_column('timestamp_year')
    merged.drop_column('timestamp_month')
    return merged

def join_perf_acq_gdfs(perf, acq, **kwargs):
    perf = null_workaround(perf)
    acq = null_workaround(acq)
    return perf.merge(acq, how='left', on=['loan_id'], type='hash')

def last_mile_cleaning(df, **kwargs):
    drop_list = [
        'loan_id', 'orig_date', 'first_pay_date', 'seller_name',
        'monthly_reporting_period', 'last_paid_installment_date', 'maturity_date', 'ever_30', 'ever_90', 'ever_180',
        'delinquency_30', 'delinquency_90', 'delinquency_180', 'upb_12',
        'zero_balance_effective_date','foreclosed_after', 'disposition_date','timestamp'
    ]
    for column in drop_list:
        df.drop_column(column)
    for col, dtype in df.dtypes.iteritems():
        if str(dtype)=='category':
            df[col] = df[col].cat.codes
        df[col] = df[col].astype('float32')
    df['delinquency_12'] = df['delinquency_12'] > 0
    df['delinquency_12'] = df['delinquency_12'].fillna(False).astype('int32')
    for column in df.columns:
        df[column] = df[column].fillna(-1)
    return df.to_arrow(preserve_index=False)

def main():
    #print('XGBOOST_BUILD_DOC is ' + os.environ['XGBOOST_BUILD_DOC'])
    parser = argparse.ArgumentParser("rapidssample")
    parser.add_argument("--data_dir", type=str, help="location of data")
    parser.add_argument("--num_gpu", type=int, help="Number of GPUs to use", default=1)
    parser.add_argument("--part_count", type=int, help="Number of data files to train against", default=2)
    parser.add_argument("--end_year", type=int, help="Year to end the data load", default=2000)
    parser.add_argument("--cpu_predictor", type=str, help="Flag to use CPU for prediction", default='False')
    parser.add_argument('-f', type=str, default='') # added for notebook execution scenarios
    args = parser.parse_args()
    data_dir = args.data_dir
    num_gpu = args.num_gpu
    part_count = args.part_count
    end_year = args.end_year
    cpu_predictor = args.cpu_predictor.lower() in ('yes', 'true', 't', 'y', '1')

    if cpu_predictor:
        print('Training with CPUs require num gpu = 1')
        num_gpu = 1

    print('data_dir = {0}'.format(data_dir))
    print('num_gpu = {0}'.format(num_gpu))
    print('part_count = {0}'.format(part_count))
    #part_count = part_count + 1 # adding one because the usage below is not inclusive
    print('end_year = {0}'.format(end_year))
    print('cpu_predictor = {0}'.format(cpu_predictor))
    
    import subprocess

    cmd = "hostname --all-ip-addresses"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    IPADDR = str(output.decode()).split()[0]
    
    cluster = LocalCUDACluster(ip=IPADDR,n_workers=num_gpu)
    client = Client(cluster)
    client
    print(client.ncores())

# to download data for this notebook, visit https://rapidsai.github.io/demos/datasets/mortgage-data and update the following paths accordingly
    acq_data_path = "{0}/acq".format(data_dir) #"/rapids/data/mortgage/acq"
    perf_data_path = "{0}/perf".format(data_dir) #"/rapids/data/mortgage/perf"
    col_names_path = "{0}/names.csv".format(data_dir) # "/rapids/data/mortgage/names.csv"
    start_year = 2000
#end_year = 2000 # end_year is inclusive -- converted to parameter
#part_count = 2 # the number of data files to train against -- converted to parameter

    client.run(initialize_rmm_pool)
    client
    print(client.ncores())
# NOTE: The ETL calculates additional features which are then dropped before creating the XGBoost DMatrix.
# This can be optimized to avoid calculating the dropped features.
    print("Reading ...")
    t1 = datetime.datetime.now()
    gpu_dfs = []
    gpu_time = 0
    quarter = 1
    year = start_year
    count = 0
    while year <= end_year:
        for file in glob(os.path.join(perf_data_path + "/Performance_" + str(year) + "Q" + str(quarter) + "*")):
            if count < part_count:
                gpu_dfs.append(process_quarter_gpu(client, col_names_path, acq_data_path, year=year, quarter=quarter, perf_file=file))
                count += 1
                print('file: {0}'.format(file))
                print('count: {0}'.format(count))
        quarter += 1
        if quarter == 5:
            year += 1
            quarter = 1
            
    wait(gpu_dfs)
    t2 = datetime.datetime.now()
    print("Reading time ...")
    print(t2-t1)
    print('len(gpu_dfs) is {0}'.format(len(gpu_dfs)))
    
    client.run(cudf._gdf.rmm_finalize)
    client.run(initialize_rmm_no_pool)
    client
    print(client.ncores())
    dxgb_gpu_params = {
        'nround':            100,
        'max_depth':         8,
        'max_leaves':        2**8,
        'alpha':             0.9,
        'eta':               0.1,
        'gamma':             0.1,
        'learning_rate':     0.1,
        'subsample':         1,
        'reg_lambda':        1,
        'scale_pos_weight':  2,
        'min_child_weight':  30,
        'tree_method':       'gpu_hist',
        'n_gpus':            1, 
        'distributed_dask':  True,
        'loss':              'ls',
        'objective':         'gpu:reg:linear',
        'max_features':      'auto',
        'criterion':         'friedman_mse',
        'grow_policy':       'lossguide',
        'verbose':           True
    }
      
    if cpu_predictor:
        print('Training using CPUs')
        dxgb_gpu_params['predictor'] = 'cpu_predictor'
        dxgb_gpu_params['tree_method'] = 'hist'
        dxgb_gpu_params['objective'] = 'reg:linear'
        
    else:
        print('Training using GPUs')
    
    print('Training parameters are {0}'.format(dxgb_gpu_params))
    
    gpu_dfs = [delayed(DataFrame.from_arrow)(gpu_df) for gpu_df in gpu_dfs[:part_count]]
    gpu_dfs = [gpu_df for gpu_df in gpu_dfs]
    wait(gpu_dfs)
    
    tmp_map = [(gpu_df, list(client.who_has(gpu_df).values())[0]) for gpu_df in gpu_dfs]
    new_map = {}
    for key, value in tmp_map:
        if value not in new_map:
            new_map[value] = [key]
        else:
            new_map[value].append(key)
    
    del(tmp_map)
    gpu_dfs = []
    for list_delayed in new_map.values():
        gpu_dfs.append(delayed(cudf.concat)(list_delayed))
    
    del(new_map)
    gpu_dfs = [(gpu_df[['delinquency_12']], gpu_df[delayed(list)(gpu_df.columns.difference(['delinquency_12']))]) for gpu_df in gpu_dfs]
    gpu_dfs = [(gpu_df[0].persist(), gpu_df[1].persist()) for gpu_df in gpu_dfs]
    
    gpu_dfs = [dask.delayed(xgb.DMatrix)(gpu_df[1], gpu_df[0]) for gpu_df in gpu_dfs]
    gpu_dfs = [gpu_df.persist() for gpu_df in gpu_dfs]
    gc.collect()
    wait(gpu_dfs)
    
    labels = None
    t1 = datetime.datetime.now()
    bst = dxgb_gpu.train(client, dxgb_gpu_params, gpu_dfs, labels, num_boost_round=dxgb_gpu_params['nround'])
    t2 = datetime.datetime.now()
    print("Training time ...")
    print(t2-t1)
    print('str(bst) is {0}'.format(str(bst)))
    print('Exiting script')

if __name__ == '__main__':
    main()
