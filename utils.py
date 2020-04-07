import pandas as pd
import numpy as np


def clean_columns(covid_data):
    covid_data.columns = [col.replace("/", "_") for col in covid_data.columns]


def plot_in_comparison_to_china(covid_data, header):
    total_numbers_with_china_df = pd.DataFrame({"%s (with China)" % (header): covid_data.transpose()[4:].sum(axis=1)})
    total_numbers_with_china_index = list(range(len(total_numbers_with_china_df)))

    covid_data_china = covid_data[covid_data["Country_Region"] == "China"]
    total_numbers_china_df = pd.DataFrame({"%s (only China)" % (header): covid_data_china.transpose()[4:].sum(axis=1)})

    covid_data_nonchina = covid_data[covid_data["Country_Region"] != "China"]
    total_numbers_df = pd.DataFrame({"%s" % (header): covid_data_nonchina.transpose()[4:].sum(axis=1)})
    total_numbers_index = list(range(len(total_numbers_df)))
    total_numbers_china_df.join(total_numbers_df).join(total_numbers_with_china_df).plot(figsize=(16, 6))
    
    return total_numbers_df


def plot_active_cases(covid_confirmed, covid_recovered, covid_died, start_date=None, country=None, filter_negate=False, ax=None):
    if country:
        comparison_sign = "!=" if filter_negate else "=="
        confirmed = covid_confirmed.query("Country_Region {} \"{}\"".format(comparison_sign, country))
        recovered = covid_recovered.query("Country_Region {} \"{}\"".format(comparison_sign, country))
        died = covid_died.query("Country_Region {} \"{}\"".format(comparison_sign, country))
    else:
        confirmed = covid_confirmed
        recovered = covid_recovered
        died = covid_died
        
    confirmed = confirmed[confirmed.columns[4:]].sum()
    recovered = recovered[recovered.columns[4:]].sum()
    died = died[died.columns[4:]].sum()
    
    for df in [confirmed, recovered, died]:
        df.index = pd.to_datetime(df.index.str.replace("_", "/"))
    
    active_cases = confirmed - died - recovered
    
    if start_date:
        active_cases = active_cases[start_date:]
    
    pd.DataFrame(data=active_cases, columns=[country if country else "Global"]).plot(figsize=(16, 6), ax=ax)


def exp(x, base=2, offset=0):
    return base ** (np.array(x) - offset)


def sigmoid(x, x0=0, lambd=1, L=1):
    return L / (1 + np.exp(-lambd * (x - x0)))


