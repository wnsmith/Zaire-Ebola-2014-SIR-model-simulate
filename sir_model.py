import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
import scipy.integrate
import sys


world_population = {
    'Guinea': {
        '2020': [13132.795, 0.00283],
        '2019': [12771.246, 0.00288],
        '2018': [12414.293, 0.00287],
        '2017': [12067.519, 0.00280],
        '2016': [11738.429, 0.00268],
        '2015': [11432.088, 0.00232],
        '2010': [10192.176, 0.00227],
        '2005': [9109.581, 0.00202],
        '2000': [8240.730, 0.00254],

    },
    'Liberia': {
        '2020': [5057.983, 0.00244],
        '2019': [4937.983, 0.00246],
        '2018': [4818.983, 0.00248],
        '2017': [4702.983, 0.00252],
        '2016': [4586.983, 0.00256],
        '2015': [4472.983, 0.00282],
        '2010': [3891.983, 0.00387],
        '2005': [3218.983, 0.00247],
        '2000': [2848.983, 0.00686],
    },
}

def get_args():
    parser = argparse.ArgumentParser(
        description="Examples:\n"
                    "$ python3 sir_model.py --simulate 1 -i 1 -a 0.1 -b 0.35 -d 100 -ps 100\n" 
                    "python3 sir_model.py --country Liberia",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-c",
        "--country",
        type=str,
        choices=['Guinea', 'Liberia'],
        help="input truth file",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--simulate",
        type=bool,
        help="input truth file",
        required=False,
    )
    parser.add_argument(
        "-ps",
        "--population_size",
        type=int,
        help="input truth file",
        required=False,
    )
    parser.add_argument(
        "-ii",
        "--initial_infectous",
        type=int,
        help="input truth file",
        required=False,
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        help="input truth file",
        required=False,
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=bool,
        help="input truth file",
        required=False,
    )
    parser.add_argument(
        "-d",
        "--days",
        type=int,
        help="input truth file",
        required=False,
    )
    args = parser.parse_args()
    return args

def validate_args(args):
    if args.simulate:
        related_args = np.array([
            args.population_size,
            args.initial_infectous,
            args.days,
            args.alpha,
            args.beta,
        ])
        if np.any(np.isnan(related_args)):
            print('Some required args for simulation mode are missing. See usage.')
            sys.exit(1)

def plot_stats(time, total_population, suspected, infectous, recovered, country = None):
    plt.figure(figsize=[6, 4])

    plt.plot(time, total_population, label='POPULATION', color = 'blue', marker='x')
    plt.plot(time, suspected, label='SUSPECTED', color = 'green', linestyle = 'dashed')
    plt.plot(time, infectous, label='INFECTOUS', color = 'red')
    plt.plot(time, recovered, label='RECOVERED', color = 'black')

    plt.grid()
    plt.legend()
    plt.xlabel("days")
    plt.xlabel("stats")

    if country:
        plt.savefig('{}.svg'.format(country))
    else:
        plt.savefig('simulation.svg')


def simulate_sir_model(population, initial_infectous, days, alpha, beta, scale = 1000):
    def SIR_model(y, t, alpha, beta):
        S, I, R = y

        dS_dt = -beta * S * I
        dI_dt = beta * S * I - alpha * I
        dR_dt = alpha * I

        return ([dS_dt, dI_dt, dR_dt])

    time = np.linspace(0, 100, 10000)
    population_arr = [population/scale]*len(time)


    s_0 = population / scale - initial_infectous / scale
    i_0 = initial_infectous / scale
    r_0 = 0

    eq_sol = scipy.integrate.odeint(SIR_model, [s_0, i_0, r_0], time, args=(alpha, beta))
    eq_sol = np.array(eq_sol)

    return time, population_arr, eq_sol[:, 0], eq_sol[:, 1], eq_sol[:, 2]

def data_preproccessing(country):
    url = 'https://raw.githubusercontent.com/cmrivers/ebola/master/country_timeseries.csv'
    response = urllib.request.urlopen(url)
    data = pd.read_csv(response)

    casses_cumulative = data.loc[:,['Date', 'Day', 'Cases_{}'.format(country)]]
    deaths_cumulative = data.loc[:,['Date', 'Deaths_{}'.format(country)]]

    per_day_stats = pd.merge(left=casses_cumulative, right=deaths_cumulative, how='inner', on='Date')
    cleaned = per_day_stats.dropna()

    return cleaned.sort_values('Day')

def get_2014_population_and_average_death_rate(world_population, country):
    country_population = world_population[country]['2015'][0]
    year_death_rates = np.array([v[1] for k,v in world_population[country].items()])
    average_death_rate = np.average(year_death_rates)

    # estimate population size in 2014
    country_population_2014 = country_population - country_population * average_death_rate
    # estimate daily mortality
    daily_moratlity_2014 = (country_population-country_population_2014)/365

    return country_population_2014, daily_moratlity_2014

def track_sir_model(data, population_size, daily_moratility):
    # initial values, naivly total population is marked as suspected
    suspected = [population_size]
    infectous = [0]
    recovered_or_death = [0]
    population = [population_size]
    time = [0]

    days = data.iloc[:,1].to_numpy()
    new_casses = data.iloc[:,2].to_numpy()
    new_deaths = data.iloc[:,3].to_numpy()

    for i in range(days.shape[0]-1):
        time.append(days[i])
        population.append(int(population_size + (daily_moratility*population_size)*(days[i+1]-days[i])))
        infectous.append(new_casses[i])
        recovered_or_death.append(new_deaths[i])
        suspected.append(population[i] - infectous[i] - recovered_or_death[i])

    return time, population, suspected, infectous, recovered_or_death

def main():
    args = get_args()
    validate_args(args)

    if args.country:
        cleaned_data = data_preproccessing(args.country)
        cp, dm = get_2014_population_and_average_death_rate(world_population, args.country)

        time, population, S, I, R = track_sir_model(cleaned_data, cp, dm)
        plot_stats(
            time=time,
            total_population=population,
            suspected=S,
            infectous=I,
            recovered=R,
            country = args.country,
        )
    if args.simulate:
        time, population, S, I, R = simulate_sir_model(
            args.population_size,
            args.initial_infectous,
            args.days, args.alpha,
            args.beta
        )
        plot_stats(
            time=time,
            total_population=population,
            suspected=S,
            infectous=I,
            recovered=R,
        )

if __name__ == "__main__":
    main()