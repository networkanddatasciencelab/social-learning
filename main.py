"""
Complete code for social learning.
"""

import os
import time
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm, skewnorm, lognorm


sns.set(style="ticks", palette="muted", color_codes=True)
options = {'font_family': 'serif', 'font_weight': 'semibold', 'font_size': '12', 'font_color': '#ffffff'}


def create_graph(net, n, e=10, prob=0.1, plot=False):
    """
    Create a network (to represent a group).
    
    Parameters
    ----------
        net : the structure of graph.
        n : the number of nodes.
        e : the number of edges linked to each node.
        prob : rewiring probability in WS networks.
        plot : whether to plot this graph.
        
    Returns
    -------
        G : a graph.
    """
    # create graph
    if net == 'FC':    # fully connected graph
        G = nx.complete_graph(n)  
        print('Fully connected (FC) network')
    elif net == 'WS':
        G = nx.generators.watts_strogatz_graph(n, e, prob)
        print('WS small-world network with rewiring probability {}'.format(prob))
    elif net == 'BA':
        G = nx.generators.barabasi_albert_graph(n, int(e/2))
        print('BA scale-free network')
    elif net == 'Lattice':   # locally connected lattice (Lattice)
        G = nx.generators.watts_strogatz_graph(n, e, p=0)     ## note that
        print('Locally connected lattice (Lattice)')
    else:
        print('Error: Network type input error!')
        return
    
    print('The number of nodes is %d' % nx.number_of_nodes(G))
    print('The number of edges is %d' % nx.number_of_edges(G))
    
    # plot graph
    if plot:
        fig = plt.figure(figsize=(10, 8), dpi=100)
        pos = nx.circular_layout(G)
        plt.axis("off")
        nx.draw_networkx_nodes(G, pos, node_size=200, edgecolors='k')
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=9, font_color='w')
        plt.show()
        
    return G


def set_sqb(G, dist, skew=0, plot=False):
    """
    Set group status quo bias (SQB) distribution.
    
    Parameters
    ----------
        G: a graph.
        dist: the distribution of group SQB.
        skew: the skewness of skewed distribution.
        plot: whether to plot this SQB distribution.
    """
    num = nx.number_of_nodes(G)
    sqb = []   # group status quo bias
    
    if dist in ['ND', 'RSND', 'LSND']:
        x = []
        if dist == 'ND':
            print('Normal distribution')
            for i in range(num):
                x.append(norm.ppf(0.01*(1+(98/(num-1)*i))))  # normal distribution
        elif dist == 'RSND': 
            print('Right skew normal distribution with skewness=%.1f' % skew)
            for i in range(num):
                x.append(skewnorm.ppf(0.001*(1+(998/(num-1)*i)), a=skew))   # right skew normal distribution
        elif dist == 'LSND': 
            print('Left skew normal distribution with skewness=%.1f' % skew)
            for i in range(num):
                x.append(skewnorm.ppf(0.001*(1+(998/(num-1)*i)), a=-skew))  # lift skew normal distribution
        x_max = max(x)
        x_min = min(x)
        y = [(i-x_min)/(x_max-x_min) for i in x]
        sqb = y                     
    else:
        print('Error: Distribution type input error!')
        return
    
    mean = np.mean(sqb)
    half = [i for i in sqb if i > 0.5]
    print("The number of sqb values over 0.5 is %.2f" % (len(half)/num))
    
    sqb_dict = {}   # status quo bias of nodes
    for i in range(num):
        sqb_dict[i] = sqb[i]
        
    nx.set_node_attributes(G, sqb_dict, 'status_quo_bias')    # set status_quo_bias as the attribute of nodes
                     
    # plot SQB distribution
    if plot:
        fig = plt.figure(figsize=(6,4), dpi=100)
        ax1 = fig.add_subplot(1, 1, 1, title='SQB distribution')  
        sns.distplot(sqb, hist=False, ax=ax1, kde_kws={'color':'k','label':'KDE'}) 
        ax2 = ax1.twinx() 
        sns.distplot(sqb, bins=10, kde=False, ax=ax2)
        txt = '\n'.join((
                r'SQB > 0.5: %.2f' % (len(half)/num),
                r'SQB mean: %.2f' % mean))
        ax1.text(0.76, 0.82, txt, fontsize=10, transform=ax1.transAxes, verticalalignment='top')
        plt.show()                        
    

def task(t, opt, power, peak):
    """
    Optimization test functions.
    
    Parameters
    ----------
        t: solution.
        opt: the type of task.
        power: the number of power.
        peak: the number of peak.
        
    Returns
    -------
        y : payoff.
    """
    if opt == 'Ackley':   # ackley function
        a = 20
        b = 12
        c = 2*peak*np.pi     # this contant is twice the number of peaks
        d = 22.30082734
        y = 1 - ((a*(1-np.exp(-b*np.sqrt((t-0.5)**2))) - np.exp(np.cos(c*(t-0.5))) + np.exp(1))/d) 
        return y**power
    elif opt == 'Rastrigin':   # Rastrigin function
        a = 1
        b = 10
        c = 2*peak*np.pi    # this contant is twice the number of peaks
        d = 4.5
        y = 1 - ((a + b*(t-0.5)**2 - a*np.cos(c*(t-0.5)))/d)
        return y**power

    
def plot_opti_func(opt, power, peak, plot=False):
    """
    Plot the optimization function
    """
    txt = opt + ' task with %d-peak under power=%d' % (peak, power)
    print(txt)
    if plot:
        t = np.arange(0.0, 1.0, 0.0001)
        s = task(t, opt, power, peak)
        fig, axs = plt.subplots(figsize=(8, 4), dpi=120)
        axs.plot(t, s)
        axs.set_xlabel('solution')
        axs.set_ylabel('performance')
        axs.grid(True, linestyle='-.')
        axs.tick_params(labelcolor='r', labelsize='medium', width=3)
        plt.title(opt + ' task')
        plt.show()


def init_solution(opt, n, power, peak, srd, prd, plot=False):
    """
    Generate initial solutions with payoff.
    
    Parameters
    ----------    
        opt: the optimization function.
        n: the number of nodes.
        power: the power number.
        peak: the number of peaks in Rastrigin function.
        srd: the round number of solution.
        prd: the round number of performance.
        
    Returns
    -------
        init_config : initial configuration.        
    """
    solution = []
    performance = []
    for i in range(n):
        solution.append(np.round(np.random.random_sample(), srd))
        performance.append(np.round(task(solution[i], opt, power, peak), prd))
    init_aver_perf = np.round(sum(performance) / n, prd)

    print('Average performance of initial solutions is %.4f' % init_aver_perf)
    print('Maximum performance of initial solutions is %.4f' % max(performance))

    init_solution = {}
    init_performance = {}
    for i in range(n):
        init_solution[i] = solution[i]
        init_performance[i] = performance[i]
    
    init_config = {}
    init_config['init_solution'] = init_solution
    init_config['init_performance'] = init_performance
    init_config['init_max_performance'] = max(performance)
    init_config['init_average_performance'] = init_aver_perf
    
    if plot == True:    
        plt.subplots(figsize=(6, 4), dpi=100)
        plt.stem(solution, performance, use_line_collection=True)
        plt.xlabel('solution')
        plt.ylabel('performance')
        plt.title('Init solution and performance')
        plt.show()        
        
    return init_config


def select_solution(group, nd, sls, ns):
    """
    Individuals select a solution from neighbors using given social learning strategy.
    
    Parameters
    ----------    
        group: a group (embedded in a graph).
        nd: an individual (a node in the graph).
        sls: the social learning strategy.
        ns: the number of sample.
        
    Returns
    -------
        pick_solu: selected solution.
        pick_perf : the performance of selected solution.        
    """
    # sample solutions of its neighbors
    solu_perf = {}
    solutions = []
    if ns > len(list(nx.neighbors(group, nd))):
        ns = len(list(nx.neighbors(group, nd)))
    nb_sample = np.random.choice(list(nx.neighbors(group, nd)), ns, replace=False)  # sample neighbors

    for nb in nb_sample:
        nb_solution = group.nodes[nb]['solution']
        nb_performance = group.nodes[nb]['performance']
        solu_perf[nb_solution] = nb_performance
        solutions.append(nb_solution)
        
    # select one solution by different social learning strategies
    pick_solu = 0  # the one solution selected from its neighbors
    pick_perf = 0  # the performance corresponding to the picked solution
    
    if sls == '0':   # Best strategy
        pick_perf = max(solu_perf.values())
        pick_solu = max(solu_perf, key=solu_perf.get)
    elif sls == '1':  # Conformity strategy
        conformity = Counter(solutions).most_common(1)
        pick_solu = conformity[0][0]
        pick_perf = solu_perf[pick_solu]
    elif sls == '2':  # Random strategy
        solu_list = np.random.choice(solutions, 1)
        pick_solu = solu_list[0]
        pick_perf = solu_perf[pick_solu]
    else:
        print('Error: Learning strategy type input error!')
        return
    
    return pick_solu, pick_perf
                
                                       
def social_learning(G, strategy, opt, power, peak, srd, prd, steps, samp, init_config):
    """
    Individuals in a group rely on social learning to search for new solutions.
    
    Parameters
    ---------- 
        G: the group.
        strategy: social learning strategies.
        opt: the optimization function.
        power: the scale of solution.
        peak: the number of peaks.
        srd: the round number of solution.
        prd: the round number of performance.
        steps: the number of time steps.
        samp: the number of sampled neighbors.
        init_config: the initial solutions and performance of the group.
    
    Returns
    ------- 
        aver_performance: the average performance of the group in each time step.
        unique_solutions: 
    
    """
    # it is necessary to give the nodes the initial solution and performance again    
    nx.set_node_attributes(G, init_config['init_solution'], 'solution')
    nx.set_node_attributes(G, init_config['init_performance'], 'performance') 
    
    aver_performance = []    # save the average performance of the group in each time step
    aver_performance.append(init_config['init_average_performance'])
    
    unique_solutions = []  # the number of unique solutions in each time step
    init_solutions = []
    for k,v in init_config['init_solution'].items():
        init_solutions.append(v)
    unique_solutions.append(len(set(init_solutions))) 

    for i in range(steps):
        one_performance = []
        one_solution = []
        for nd in G.nodes():
            nd_perf = G.nodes[nd]['performance']
            nd_solu = G.nodes[nd]['solution']
            nd_sqb = G.nodes[nd]['status_quo_bias']
                     
            if strategy in ['0', '1', '2']:
                pick_solu, pick_perf = select_solution(G, nd, strategy, samp)   
            else:
                raise ValueError
            
            # social learning or not
            if pick_perf > nd_perf:
                new_solu = np.round((1 - nd_sqb) * pick_solu + nd_sqb * nd_solu, srd)
                G.nodes[nd]['solution'] = new_solu
                new_perf = np.round(task(new_solu,opt,power,peak), prd)
                G.nodes[nd]['performance'] = new_perf
                
            one_performance.append(G.nodes[nd]['performance'])
            one_solution.append(G.nodes[nd]['solution'])
            
        aver_perf = np.round(sum(one_performance) / G.number_of_nodes(), prd)  # group average performance
        aver_performance.append(aver_perf)
        unique_solutions.append(len(set(one_solution))) 
        # print('The average performance of this iteration is %5.4f' %  aver_perf)
        
    return aver_performance, unique_solutions


if __name__ == '__main__':
    
    # All parameters
    trials = 10      # Experiment number--the number of independent experiments
    net = 'FC'       # Network structure--Fully Connected('FC'), 'WS', 'BA', 'Lattice'
    n = 100          # Group size--the number of nodes
    e = 10           # Node degree--the number of edges linked to a node
    prob = 0         # Rewirng probality in WS network
    dist = 'ND'      # SQB distribution--['ND', 'RSND', 'LSND']
    skew = 0         # Skewness--the degree of skew distribution
    opt = 'Ackley'   # Task environment--Ackley (hard-to-easy), Rastrigin (easy-to-hard)
    power = 1        # Compress ratio--the power number of solutions (0-8)
    peak = 1         # the number of peaks in Rastrigin function 
    srd = 1          # the round number of solution (task granularity).
    prd = 4          # the round number of performance.
    samp = 3         # Neighbor sampling--the number of sampled neighbors in SLS (0-n)
    steps = 500      # Learning time steps--the number of iterations
    plot = False     # plot or not
    
    LS = {'0': 'Best', '1': 'Conformity', '2': 'Random'}   # social learning strategies
    Distribution = ['ND', 'RSND', 'LSND']
    Network = ['FC', 'WS', 'BA', 'Lattice']
    Skewness = {'0': 50, '1.1': 40, '-1.1': 60, '1.75': 30, '-1.75': 70, '3': 20, '-3': 80, '60': 10, '-60': 90}  # skewness and proportion
    Peaks = [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383]
    
    path = './results/test/'    # path to save results
    
    for srd in range(1, 11):
        for peak in Peaks:
            proportion = Skewness[str(skew)]
            if skew == 0:
                dist = 'ND'
            elif skew > 0:
                dist = 'RSND'
                proportion = 50
            elif skew < 0:
                dist = 'LSND'
                skew = abs(skew)

            print('************* Start new experiment ***************')
            output = path + '/peak' + str(peak) + '/srd' + str(srd) + '/'   ## save path
            if not os.path.exists(output):
                os.makedirs(output)
                print('Directory created successfully!')
            print(output)
            print('\n')
            
            for opt in ['Ackley', 'Rastrigin']:
                if opt == 'Rastrigin':
                    power = 8
                    
                print('---------------- Start a new task -----------------')
                print('trials=%d, proportion=%s, srd=%d, peak=%d, power=%d, steps=%d' % (trials, proportion, srd, peak, power, steps))
                print('\n')
                
                results = []
                for i in range(trials):
                    print('######### Trial %d #########' % i)
                    G = create_graph(net, n, e, prob, plot=False)    # create a network
                    edges = nx.number_of_edges(G)              # the number of edges in this network
                    set_sqb(G, dist, skew, plot)               # set group status quo bias
                    # plot_opti_func(opt, power, peak, plot)     # plot task environment
                    init_config = init_solution(opt, n, power, peak, srd, prd, plot)  
                    
                    # get initial solutions and performance
                    s_time = time.time()
                    for strategy_key, strategy_name in LS.items():
                        print('--------------------------------------------------')
                        print(strategy_name)
                        sta_time = time.time()
                        performance, solution = social_learning(G, strategy_key, opt, power, peak, srd, prd, steps, samp, init_config)  
                        end_time = time.time()
                        run_time = round(end_time - sta_time, 2)
                        print('@Runing Time of this strategy is %.2f seconds' % run_time)
                        res = [trials, i+1, net, n, edges, prob, dist, skew, proportion, opt, peak, power,
                               srd, prd, samp, strategy_name, steps, run_time, init_config['init_max_performance']]
                        res.extend(performance)
                        res.extend(solution)
                        results.append(res)
                        
                    e_time = time.time()
                    print('--------------------------------------------------')
                    print('@Runing Time of this trial is %.2f seconds' % (e_time - s_time))
                    print('\n')
                    
                print('------------- Completed this task ----------------')    
                res_df = pd.DataFrame(results)
                columns_dict = {0:'trials', 1:'trial', 2: 'network', 3:'nodes', 4: 'edges',  
                                5:'probability', 6:'distribution', 7:'skewness', 8: 'proportion', 9: 'task', 
                                10:'peak', 11:'power', 12:'srd', 13:'prd', 14:'samples',  
                                15:'strategy', 16: 'steps', 17: 'time', 18:'init_max_performance'}
                res_df.rename(columns=columns_dict, inplace=True)  # rename columns
                col_dict = {}
                for i in range(2*steps+2):
                    if i < steps+1:
                        col_dict.update({i+19: 'p' + str(i)})
                    else:
                        col_dict.update({i+19: 's' + str(i-steps-1)})
                res_df.rename(columns=col_dict, inplace=True)
                res_file = output + opt + '_task.csv'
                res_df.to_csv(res_file, index=False)
                print(res_file)
                print('*********** Completed this task ************')
                print('\n')
                
            print('*********** Completed this peak ************')
            print('\n')
            
        print('*********** Completed this granularity ************')
        print('\n')
            
    print('～～～～～～ Completed all experiments ～～～～～～')