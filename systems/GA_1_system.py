import numpy as np
import pandas as pd
import random
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
        # calculate profit
        profit = [CLOSE[-1,market]-CLOSE[-2,market] for market in range(nMarkets)]

        # fitness function is total portfolio value after taking the value
        diff = np.nansum([market*pos for market,pos in zip(profit, gen)])
        
        return diff 
        
    def selection(pop, scores, tournament_size=tournament_size):
        rand_selection = np.random.randint(len(pop))
        for s in np.random.randint(0,len(pop), tournament_size-1):
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
                # change the bit
                poss_values = [-1,0,1]
                poss_values.remove(bitstring[i])
                bitstring[i] = random.choice(poss_values)
  

    pop = [np.random.randint(-1, 2, size = nMarkets).tolist() for _ in range(n_pop)]
    best, best_eval = pop[0], fitness(pop[0])


    for gen in range(n_iter):
        scores = [fitness(gen) for gen in pop]
        for i in range(n_pop):
            if scores[i] > best_eval:
                best = pop[i]
                best_eval = scores[i]
        
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
        print(f"At iter {gen}, The best score is {best_eval}")
            

    best = np.array(best)
    #best =  best/np.nansum(abs(best))
    return best, settings

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

    settings['population_size'] = 100
    settings['tournament_size'] = 5
    settings['crossover_rate'] = 0.3
    settings['mutation_rate'] = 0.02
    settings['n_iter'] = 100


    return settings

if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)