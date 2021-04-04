import numpy as np
import statistics
import random
import talib
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
print(num_cores)

np.random.seed(42)
random.seed(42)


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    '''This system uses trend following techniques to allocate capital into the desired equities.'''
    nMarkets= len(settings['markets'])
    lookback = settings['lookback']
    n_pop = settings['population_size']
    tournament_size = settings['tournament_size']
    crossover_rate = settings['crossover_rate']
    mutation_rate = settings['mutation_rate']
    n_iter = settings['n_iter']
    init_budget = settings['budget']


    def fitness(gen):
        """
        params: the generation whose fitness we want to evaluate
        best: the previous generation

        """
        ''' fitness 1
        # use the sharpe ratio as fitness
        returns = []
        
        for i in range(lookback-1):
            prev = np.nansum([CLOSE[i,market] * pos for market, pos in zip(range(nMarkets), gen)])
            pred = np.nansum([CLOSE[i+1,market] * pos for market, pos in zip(range(nMarkets), gen)])

            percentage_chg = (pred - prev) / prev
            returns.append(percentage_chg)
        

        return statistics.mean(returns) /statistics.stdev(returns)
        '''
        
        # get expected returns from signals
        signals = []

        for market in range(nMarkets):           
            prev_close = CLOSE[:,market]
            prev_high = HIGH[:,market]
            prev_low = LOW[:,market]
            roc = talib.ROC(prev_close)
            rsi = talib.RSI(prev_close)
            fastk, fastd = talib.STOCHF(prev_high, prev_low, prev_close)

            # normalize to 0,1 range
            roc_val = (roc[-1] - min(roc)) / (max(roc) - min(roc))
            rsi_val = (rsi[-1] - min(rsi)) / (max(rsi) - min(rsi))
            macd_val = pd.DataFrame(CLOSE[:,market]).ewm(span=12,adjust=False).mean() -\
                        pd.DataFrame(CLOSE[:,market]).ewm(span=26,adjust=False).mean()
            signal = macd_val - macd_val.ewm(span=9, adjust=False).mean()

            # if the signal persist for 3 day
            macd_val =(signal[-3:].mean()[0] - min(signal)) / (signal.max() - signal.min())
            
            fastk_val =(fastk[-1] - min(fastk)) / (max(fastk) - min(fastk))
            fastd_val = (fastd[-1] - min(fastd)) / (max(fastd) - min(fastd))
            
            if(roc_val < gen[0][0]\
                and rsi_val < gen[1][0] \
                and macd_val < gen[2][0] \
                and fastk_val < gen[3][0] \
                and fastd_val < gen[4][0]):
                signals.append(1)
            elif (roc_val < gen[0][1]\
                and rsi_val < gen[1][1] \
                and macd_val < gen[2][1] \
                and fastk_val < gen[3][1] \
                and fastd_val < gen[4][1]):
                signals.append(0)
            else:
                signals.append(-1)
        
        gains = []
        for signal in range(len(signals)):
            prev_day = CLOSE[-2, signal]
            curr_day = CLOSE[-1, signal]
            gain = (curr_day - prev_day) * -signals[signal] / prev_day

            gains.append(gain)
        expected_return = np.nansum(gains)        
        return expected_return, signals


    def selection(pop, scores, tournament_size=tournament_size):
        rand_selection = np.random.randint(len(pop))
        for s in np.random.randint(0,len(pop), tournament_size):
            if scores[s] > scores[rand_selection]:
                rand_selection = s
        return pop[rand_selection]

    def crossover(p1, p2, r_cross=crossover_rate):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if np.random.rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = np.random.randint(1, len(p1)-2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]
    
    def mutation(bitstring, r_mut = mutation_rate):
        for i in range(len(bitstring)):
            # check for a mutation
            if np.random.rand() < r_mut:
                # get new random values for both the upper and lower bound
                bitstring[i] = get_bounds(0,1)

                
    def get_bounds(low, high):
        higher_bound = np.random.uniform(low=low,high=high)
        lower_bound = np.random.uniform(low=low,high=higher_bound)
        return (lower_bound,higher_bound)

    pop = []
    for i in range(n_pop):
        gen = []
        for j in range(5):
            # since all our indicators have been scaled to 0,1
            gen.append(get_bounds(0,1))


        pop.append(gen)

    best, best_eval, best_signals = pop[0], fitness(pop[0])[0], fitness(pop[0])[1]

    counter = 0
    for gen in range(n_iter):
        scores = Parallel(n_jobs = num_cores)(delayed(fitness)(gen) for gen in pop)

        for i in range(n_pop):
            if scores[i][0] > best_eval:
                best = pop[i]
                best_eval = scores[i][0]
                best_signals = scores[i][1]
        
        # select the parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # use the parents to create children
        children = []
        for i in range(0,n_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            # crossover 
            cross = crossover(p1, p2)
            # mutation
            for child in cross:
                mutation(child)
                children.append(child)

        # the population become the children
        pop = children
        print(f"DATE: {DATE[counter]}: At iter {gen}, The best score is {best_eval}")
    file = open("best_gen.txt", 'a')
    file.write(f"\nDATE: {DATE[counter]}: best population: {best}, score: {best_eval}")
    file.flush()
            

    # best = np.array(best)

    best_signals = np.array(best_signals)
    return best_signals, settings

def mySettings():
    '''Define your trading system settings here.'''

    settings= {}

    # Futures Contracts
    settings['markets']= ['F_AD', 'F_C', 'F_DX', 'F_ED', 'F_ES', 'F_FC', 'F_HG', 'F_LB', 'F_LC', 'F_MD', 'F_NG', 'F_NQ', 'F_NR',
                'F_O', 'F_PA', 'F_PL', 'F_RB', 'F_RU', 'F_SB', 'F_TU', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_UB', 'F_LX',
                'F_SS', 'F_DL', 'F_ZQ', 'F_VX', 'F_BG', 'F_LU', 'F_AH', 'F_DZ', 'F_FL', 'F_FM', 'F_FY', 'F_GX', 'F_HP',
                'F_LR', 'F_LQ', 'F_NY', 'F_RF', 'F_SH', 'F_SX', 'F_EB', 'F_VW', 'F_GD', 'F_F']
    
    settings['beginInSample']= '20180101'
    settings['endInSample']= '20201231'
    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05

    settings['population_size'] = 30
    settings['tournament_size'] = 6
    settings['crossover_rate'] = 0.2
    settings['mutation_rate'] = 0.02
    settings['n_iter'] = 50

    return settings

if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)