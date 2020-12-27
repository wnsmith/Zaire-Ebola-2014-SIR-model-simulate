import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
import scipy.integrate
import sys
from sklearn import preprocessing


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
        description="SIR Model simulator for Zaire ebola 2014 diesease\n"
                    "-------------------- MODES --------------------------\n"
                    "1) classic simulator\n"
                    "   Example: $ python3 sir_model.py --simulate 1 -i 1 -a 0.1 -b 0.35 -d 100 -ps 100\n" 
                    "2) real data tracker\n"
                    "   Example: $ python3 sir_model.py --country Liberia\n"
                    "3) 1) + 2)\n"
                    "   Example: $ python3 sir_model.py --country Liberia --simulate 1 -b 0.27 -a 0.2\n"
                    "-----------------------------------------------------\n",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-c",
        "--country",
        type=str,
        choices=['Guinea', 'Liberia'],
        help="choose one of two supported countries",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--simulate",
        type=bool,
        help="indicator for simulation run",
        required=False,
    )
    parser.add_argument(
        "-ps",
        "--population_size",
        type=int,
        help="total number of samples",
        required=False,
    )
    parser.add_argument(
        "-ii",
        "--initial_infectous",
        type=int,
        help="number of initial infected samples",
        required=False,
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        help="recovery rate",
        required=False,
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=bool,
        help="disease transmission rate",
        required=False,
    )
    parser.add_argument(
        "-d",
        "--days",
        type=int,
        help="number of days for time series",
        required=False,
    )
    args = parser.parse_args()
    return args

def validate_args(args):
    if args.simulate and args.country:
        if args.alpha is None:
            print('Missing disease transmission rate (--beta/-b)\n')
        if args.beta is None:
            print('Missing recovery rate (--alpha/-a)\n')
        print('Values for [initial infectous, days, population size] will be ignored in this run.')
        return

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

def plot_stats(time, total_population, suspected, infectous, recovered, output):
    def plot_population_and_suspected():
        fig = plt.figure(figsize=[6, 4])

        plt.plot(time, total_population, label='POPULATION', color='blue', marker='x')
        plt.plot(time, suspected, label='SUSPECTED', color='green', linestyle='dashed')

        plt.grid()
        plt.legend()
        plt.xlabel("days")
        plt.ylabel("population / suspected")

        plt.savefig("{}_population.png".format(output))

    def plot_infectious_and_recovered():
        fig = plt.figure(figsize=[6, 4])

        plt.plot(time, infectous, label='INFECTIOUS', color='red')
        plt.plot(time, recovered, label='RECOVERED', color='black')

        plt.grid()
        plt.legend()
        plt.xlabel("days")
        plt.ylabel("infectious / recovered or death")

        plt.savefig("{}_inf.png".format(output))

    plot_population_and_suspected()
    plot_infectious_and_recovered()

def simulate_sir_model(population, initial_infectous, alpha, beta, scale = 1000, days = None):
    def SIR_model(y, time, beta, alpha):
        S, I, R = y

        dS_dt = -beta * S * I
        dI_dt = beta * S * I - alpha * I
        dR_dt = alpha * I

        return ([dS_dt, dI_dt, dR_dt])

    time = [x for x in range(days)] if days else np.linspace(0, 100, 100)
    population_arr = [population / scale]*len(time)


    s_0 = population / scale
    i_0 = initial_infectous / scale
    r_0 = 0.0

    eq_sol = scipy.integrate.odeint(SIR_model, [s_0, i_0, r_0], time, args=(beta, alpha))
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
    country_population = world_population[country]['2015'][0]*1000
    year_death_rates = np.array([v[1]*1000 for k,v in world_population[country].items()])
    average_death_rate = np.average(year_death_rates)

    # estimate population size in 2014
    country_population_2014 = int(country_population - country_population * average_death_rate/100)
    # estimate daily mortality
    daily_moratlity_2014 = int((country_population-country_population_2014)/365)
    return country_population_2014, daily_moratlity_2014

def track_sir_model(data, population_size, daily_moratility):
    # initial values, naivly total population is marked as suspected
    suspected = [population_size]
    infectious = [0]
    recovered_or_death = [0]
    population = [population_size]
    time = [0]

    days = data.iloc[:,1].to_numpy()
    new_casses = data.iloc[:,2].to_numpy()
    new_deaths = data.iloc[:,3].to_numpy()

    for i in range(days.shape[0]-1):
        time.append(days[i])
        population.append(population_size + daily_moratility)
        infectious.append(new_casses[i])
        recovered_or_death.append(new_deaths[i])
        suspected.append((population[i] - infectious[i] - recovered_or_death[i]))


    return time, population, suspected, infectious, recovered_or_death

def main():
    args = get_args()
    validate_args(args)

    if args.country and args.simulate:
        print("Running mode 3)\n")
        cleaned_data = data_preproccessing(args.country)
        cp, dm = get_2014_population_and_average_death_rate(world_population, args.country)

        time_x, population, S, I, R = track_sir_model(cleaned_data, cp, dm)
        plot_stats(
            time=time_x,
            total_population=population,
            suspected=S,
            infectous=I,
            recovered=R,
            output=args.country,
        )

        time, population, S, I, R = simulate_sir_model(
            population=cp,
            initial_infectous=cleaned_data['Cases_{}'.format(args.country)].to_numpy()[0] / 1000,
            alpha=args.alpha,
            beta=args.beta,
            days=time_x[len(time_x) - 1],
            scale=1,
        )
        plot_stats(
            time=time,
            total_population=population,
            suspected=S,
            infectous=I,
            recovered=R,
        )
    else:
        if args.country:
            print("Running mode 2)\n")
            cleaned_data = data_preproccessing(args.country)
            cp, dm = get_2014_population_and_average_death_rate(world_population, args.country)

            time, population, S, I, R = track_sir_model(cleaned_data, cp*1000, dm*1000)
            plot_stats(
                time=time,
                total_population=population,
                suspected=S,
                infectous=I,
                recovered=R,
                output=args.country,
            )
        if args.simulate:
            print("Running mode 1)\n")
            time, population, S, I, R = simulate_sir_model(
                population=args.population_size,
                initial_infectous=args.initial_infectous,
                alpha=args.alpha,
                beta=args.beta
            )
            plot_stats(
                time=time,
                total_population=population,
                suspected=S,
                infectous=I,
                recovered=R,
                output='simulate',
            )

if __name__ == "__main__":
    main()