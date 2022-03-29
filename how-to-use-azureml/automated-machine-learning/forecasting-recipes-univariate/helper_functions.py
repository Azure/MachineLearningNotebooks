"""
Helper functions to determine AutoML experiment settings for forecasting.
"""
import pandas as pd
import statsmodels.tsa.stattools as stattools
from arch import unitroot
from azureml.automl.core.shared import constants
from azureml.automl.runtime.shared.score import scoring


def adf_test(series, **kw):
    """
    Wrapper for the augmented Dickey-Fuller test. Allows users to set the lag order.

    :param series: series to test
    :return: dictionary of results
    """
    if "lags" in kw.keys():
        msg = "Lag order of {} detected. Running the ADF test...".format(
            str(kw["lags"])
        )
        print(msg)
        statistic, pval, critval, resstore = stattools.adfuller(
            series, maxlag=kw["lags"], autolag=kw["autolag"], store=kw["store"]
        )
    else:
        statistic, pval, critval, resstore = stattools.adfuller(
            series, autolag=kw["IC"], store=kw["store"]
        )

    output = {
        "statistic": statistic,
        "pval": pval,
        "critical": critval,
        "resstore": resstore,
    }
    return output


def kpss_test(series, **kw):
    """
    Wrapper for the KPSS test. Allows users to set the lag order.

    :param series: series to test
    :return: dictionary of results
    """
    if kw["store"]:
        statistic, p_value, critical_values, rstore = stattools.kpss(
            series, regression=kw["reg_type"], nlags=kw["lags"], store=kw["store"]
        )
    else:
        statistic, p_value, lags, critical_values = stattools.kpss(
            series, regression=kw["reg_type"], nlags=kw["lags"]
        )
    output = {
        "statistic": statistic,
        "pval": p_value,
        "critical": critical_values,
        "lags": rstore.lags if kw["store"] else lags,
    }

    if kw["store"]:
        output.update({"resstore": rstore})
    return output


def format_test_output(test_name, test_res, H0_unit_root=True):
    """
    Helper function to format output. Return a dictionary with specific keys. Will be used to
    construct the summary data frame for all unit root tests.

    TODO: Add functionality of choosing based on the max lag order specified by user.

    :param test_name: name of the test
    :param test_res: object that contains corresponding test information. Can be None if test failed.
    :param H0_unit_root: does the null hypothesis of the test assume a unit root process? Some tests do (ADF),
                         some don't (KPSS).
    :return: dictionary of summary table for all tests and final decision on stationary vs non-stationary.
             If test failed (test_res is None), return empty dictionary.
    """
    # Check if the test failed by trying to extract the test statistic
    if test_name in ("ADF", "KPSS"):
        try:
            test_res["statistic"]
        except BaseException:
            test_res = None
    else:
        try:
            test_res.stat
        except BaseException:
            test_res = None

    if test_res is None:
        return {}

    # extract necessary information
    if test_name in ("ADF", "KPSS"):
        statistic = test_res["statistic"]
        crit_val = test_res["critical"]["5%"]
        p_val = test_res["pval"]
        lags = test_res["resstore"].usedlag if test_name == "ADF" else test_res["lags"]
    else:
        statistic = test_res.stat
        crit_val = test_res.critical_values["5%"]
        p_val = test_res.pvalue
        lags = test_res.lags

    if H0_unit_root:
        H0 = "The process is non-stationary"
        stationary = "yes" if p_val < 0.05 else "not"
    else:
        H0 = "The process is stationary"
        stationary = "yes" if p_val > 0.05 else "not"

    out = {
        "test_name": test_name,
        "statistic": statistic,
        "crit_val": crit_val,
        "p_val": p_val,
        "lags": int(lags),
        "stationary": stationary,
        "Null Hypothesis": H0,
    }
    return out


def unit_root_test_wrapper(series, lags=None):
    """
    Main function to run multiple stationarity tests. Runs five tests and returns a summary table + decision
    based on the majority rule. If the number of tests that determine a series is stationary equals to the
    number of tests that deem it non-stationary, we assume the series is non-stationary.
        * Augmented Dickey-Fuller (ADF),
        * KPSS,
        * ADF using GLS,
        * Phillips-Perron (PP),
        * Zivot-Andrews (ZA)

    :param lags: (optional) parameter that allows user to run a series of tests for a specific lag value.
    :param series: series to test
    :return: dictionary of summary table for all tests and final decision on stationary vs nonstaionary
    """
    # setting for ADF and KPSS tests
    adf_settings = {"IC": "AIC", "store": True}

    kpss_settings = {"reg_type": "c", "lags": "auto", "store": True}

    arch_test_settings = {}  # settings for PP, ADF GLS and ZA tests
    if lags is not None:
        adf_settings.update({"lags": lags, "autolag": None})
        kpss_settings.update({"lags:": lags})
        arch_test_settings = {"lags": lags}
    # Run individual tests
    adf = adf_test(series, **adf_settings)  # ADF test
    kpss = kpss_test(series, **kpss_settings)  # KPSS test
    pp = unitroot.PhillipsPerron(series, **arch_test_settings)  # Phillips-Perron test
    adfgls = unitroot.DFGLS(series, **arch_test_settings)  # ADF using GLS test
    za = unitroot.ZivotAndrews(series, **arch_test_settings)  # Zivot-Andrews test

    # generate output table
    adf_dict = format_test_output(test_name="ADF", test_res=adf, H0_unit_root=True)
    kpss_dict = format_test_output(test_name="KPSS", test_res=kpss, H0_unit_root=False)
    pp_dict = format_test_output(
        test_name="Philips Perron", test_res=pp, H0_unit_root=True
    )
    adfgls_dict = format_test_output(
        test_name="ADF GLS", test_res=adfgls, H0_unit_root=True
    )
    za_dict = format_test_output(
        test_name="Zivot-Andrews", test_res=za, H0_unit_root=True
    )

    test_dict = {
        "ADF": adf_dict,
        "KPSS": kpss_dict,
        "PP": pp_dict,
        "ADF GLS": adfgls_dict,
        "ZA": za_dict,
    }
    test_sum = pd.DataFrame.from_dict(test_dict, orient="index").reset_index(drop=True)

    # decision based on the majority rule
    if test_sum.shape[0] > 0:
        ratio = test_sum[test_sum["stationary"] == "yes"].shape[0] / test_sum.shape[0]
    else:
        ratio = 1  # all tests fail, assume the series is stationary

    # Majority rule. If the ratio is exactly 0.5, assume the series in non-stationary.
    stationary = "YES" if (ratio > 0.5) else "NO"

    out = {"summary": test_sum, "stationary": stationary}
    return out


def ts_train_test_split(df_input, n, time_colname, ts_id_colnames=None):
    """
    Group data frame by time series ID and split on last n rows for each group.

    :param df_input: input data frame
    :param n: number of observations in the test set
    :param time_colname: time column
    :param ts_id_colnames: (optional) list of grain column names
    :return train and test data frames
    """
    if ts_id_colnames is None:
        ts_id_colnames = []
    ts_id_colnames_original = ts_id_colnames.copy()
    if len(ts_id_colnames) == 0:
        ts_id_colnames = ["Grain"]
        df_input[ts_id_colnames[0]] = "dummy"
    # Sort by ascending time
    df_grouped = df_input.sort_values(time_colname).groupby(
        ts_id_colnames, group_keys=False
    )
    df_head = df_grouped.apply(lambda dfg: dfg.iloc[:-n])
    df_tail = df_grouped.apply(lambda dfg: dfg.iloc[-n:])
    # drop group column name if it was not originally provided
    if len(ts_id_colnames_original) == 0:
        df_head.drop(ts_id_colnames, axis=1, inplace=True)
        df_tail.drop(ts_id_colnames, axis=1, inplace=True)
    return df_head, df_tail


def compute_metrics(fcst_df, metric_name=None, ts_id_colnames=None):
    """
    Calculate metrics per grain.

    :param fcst_df: forecast data frame. Must contain 2 columns: 'actual_level' and 'predicted_level'
    :param metric_name: (optional) name of the metric to return
    :param ts_id_colnames: (optional) list of grain column names
    :return: dictionary of summary table for all tests and final decision on stationary vs nonstaionary
    """
    if ts_id_colnames is None:
        ts_id_colnames = []
    if len(ts_id_colnames) == 0:
        ts_id_colnames = ["TS_ID"]
        fcst_df[ts_id_colnames[0]] = "dummy"
    metrics_list = []
    for grain, df in fcst_df.groupby(ts_id_colnames):
        try:
            scores = scoring.score_regression(
                y_test=df["actual_level"],
                y_pred=df["predicted_level"],
                metrics=list(constants.Metric.SCALAR_REGRESSION_SET),
            )
        except BaseException:
            msg = "{}: metrics calculation failed.".format(grain)
            print(msg)
            scores = {}
        one_grain_metrics_df = pd.DataFrame(
            list(scores.items()), columns=["metric_name", "metric"]
        ).sort_values(["metric_name"])
        one_grain_metrics_df.reset_index(inplace=True, drop=True)
        if len(ts_id_colnames) < 2:
            one_grain_metrics_df["grain"] = ts_id_colnames[0]
        else:
            one_grain_metrics_df["grain"] = "|".join(list(grain))

        metrics_list.append(one_grain_metrics_df)
    # collect into a data frame
    grain_metrics = pd.concat(metrics_list)
    if metric_name is not None:
        grain_metrics = grain_metrics.query("metric_name == @metric_name")
    return grain_metrics
