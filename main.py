import configparser
import time

from utils import Utils, Log, GPUTools
from population import initialize_population
from evaluate import decode, fitnessEvaluate
from evolve import FDE
import copy, os
from datetime import datetime


def create_directory():
    dirs = ['./log', './populations', './scripts', './trained_models']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def fitness_evaluate(population, curr_gen, idx):
    filenames = []
    if idx:
        filename = decode(population[0], curr_gen, idx)
        filenames.append(filename)
    else:
        for i, individual in enumerate(population):
            filename = decode(individual, curr_gen, i)
            filenames.append(filename)
    acc_set, num_parameters, flops = fitnessEvaluate(filenames, curr_gen, is_test=False)

    return acc_set, num_parameters, flops


def save_record(_str, first_time):
    dt = datetime.now()
    dt.strftime('%Y-%m-%d %H:%M:%S')
    if first_time:
        file_mode = 'w'
    else:
        file_mode = 'a+'
    f = open('./populations/pop_update.txt', file_mode)
    f.write('[%s]-%s\n' % (dt, _str))
    f.flush()
    f.close()


def evolve(population, acc_set, num_parameters, flops, params):
    new_pop = []
    acc_set_new = []
    num_parameters_set_new = []
    flops_set_new = []

    offspring_new = 0
    for i, individual in enumerate(population):
        individual_ = copy.deepcopy(individual)
        fde = FDE(individual_, i, population, acc_set, params, Log)
        indi_mut = fde.mutate()
        indi_new = fde.crossover(indi_mut)
        print('indi_new' + str(indi_new))
        acc_new, num_parameters_new, flops_new = fitness_evaluate([indi_new], params['gen_no'], i)

        if acc_new >= acc_set[i]:
            new_pop.append(indi_new)
            acc_set_new.append(acc_new[0])
            num_parameters_set_new.append(num_parameters_new[0])
            flops_set_new.append(flops_new[0])
            population[i] = indi_new
            offspring_new += 1
        else:
            new_pop.append(population[i])
            acc_set_new.append(acc_set[i])
            num_parameters_set_new.append(num_parameters[i])
            flops_set_new.append(flops[i])

    _str = 'EVOLVE[%d-gen]-%d offspring are generated, %d enter into next generation' % (
        params['gen_no'], len(new_pop), offspring_new)
    Log.info(_str)
    if params['gen_no'] <= 1:
        save_record(_str, first_time=True)
    else:
        save_record(_str, first_time=False)
    return new_pop, acc_set_new, num_parameters_set_new, flops_set_new


def update_best_individual(population, acc_set, num_parameters, flops, gbest):
    fitnessSet = [
        acc_set[i] * pow(num_parameters[i] / Tp, wp[int(bool(num_parameters[i] > Tp))]) * pow(flops[i] / Tf,
                                                                                              wf[
                                                                                                  int(bool(
                                                                                                      flops[
                                                                                                          i] > Tf))])
        for i in range(len(population))]
    if not gbest:
        pbest_individuals = copy.deepcopy(population)
        pbest_accSet = copy.deepcopy(acc_set)
        pbest_params = copy.deepcopy(num_parameters)
        pbest_flops = copy.deepcopy(flops)
        gbest_individual, gbest_acc, gbest_params, gbest_flops = getGbest(
            [pbest_individuals, pbest_accSet, pbest_params, pbest_flops])
    else:
        gbest_individual, gbest_acc, gbest_params, gbest_flops = gbest
        for i, acc in enumerate(acc_set):
            if acc > gbest_acc:
                gbest_individual = copy.deepcopy(population[i])
                gbest_acc = copy.deepcopy(acc)
                gbest_params = copy.deepcopy(num_parameters[i])
                gbest_flops = copy.deepcopy(flops[i])

    return [gbest_individual, gbest_acc, gbest_params, gbest_flops]


def getGbest(pbest):
    pbest_individuals, pbest_accSet, pbest_params, pbest_flops = pbest
    gbest_acc = 0
    gbest_params = 1e9
    gbest = None
    gbest_flops = 1e9
    for i, indi in enumerate(pbest_individuals):
        if pbest_accSet[i] > gbest_acc:
            gbest = copy.deepcopy(indi)
            gbest_acc = copy.deepcopy(pbest_accSet[i])
            gbest_params = copy.deepcopy(pbest_params[i])
            gbest_flops = copy.deepcopy(pbest_flops[i])
    return gbest, gbest_acc, gbest_params, gbest_flops


def fitness_test(gbest_individual):
    filename = decode(gbest_individual, -1, -1)
    acc_set, num_parameters, flops = fitnessEvaluate([filename], -1, is_test=True)
    return acc_set[0], num_parameters[0], flops[0]


def evolveCNN(params):
    gen_no = 0
    Log.info('Initialize...')
    start = time.time()
    population = initialize_population(params)
    # print(population)
    Log.info('EVOLVE[%d-gen]-Begin evaluate the fitness' % (gen_no))
    acc_set, num_parameters, flops = fitness_evaluate(population, gen_no, None)
    Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))

    # gbest
    [gbest_individual, gbest_acc, gbest_params, gbest_flops] = update_best_individual(population, acc_set,
                                                                                      num_parameters, flops,
                                                                                      gbest=None)
    Log.info('EVOLVE[%d-gen]-Finish the updating' % (gen_no))

    Utils.save_population_and_acc('population', population, acc_set, num_parameters, flops, gen_no)
    Utils.save_population_and_acc('gbest', [gbest_individual], [gbest_acc], [gbest_params], [gbest_flops], gen_no)

    gen_no += 1
    velocity_set = []
    for ii in range(len(population)):
        velocity_set.append([0] * len(population[ii]))

    for curr_gen in range(gen_no, params['num_iteration']):
        params['gen_no'] = curr_gen

        Log.info('EVOLVE[%d-gen]-Begin differential evolution' % (curr_gen))
        population, acc_set, num_parameters, flops = evolve(population, acc_set, num_parameters, flops, params)
        Log.info('EVOLVE[%d-gen]-Finish differential evolution' % (curr_gen))

        [gbest_individual, gbest_acc, gbest_params, gbest_flops] = update_best_individual(population, acc_set,
                                                                                          num_parameters, flops,
                                                                                          gbest=[gbest_individual,
                                                                                                 gbest_acc,
                                                                                                 gbest_params,
                                                                                                 gbest_flops])
        Log.info('EVOLVE[%d-gen]-Finish the updating' % (curr_gen))

        Utils.save_population_and_acc('population', population, acc_set, num_parameters, flops, curr_gen)
        Utils.save_population_and_acc('gbest', [gbest_individual], [gbest_acc], [gbest_params], [gbest_flops], curr_gen)
    
    end = time.time()
    Log.info('Total Search Time: %.2f seconds' % (end - start))
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    Log.info("%02dh:%02dm:%02ds" % (h, m, s))

    # final training and test on testset
    print('final training and test on testset, GBEST CNN architecture is' + str(gbest_individual))
    gbest_acc, num_parameters, flops = fitness_test(gbest_individual)
    Log.info('The acc of the best searched CNN architecture is [%.5f], number of parameters is [%d], flops is [%d]' % (
        gbest_acc, num_parameters, flops))
    Utils.save_population_and_acc('final_gbest', [gbest_individual], [gbest_acc], [num_parameters], [flops], -1)


def __read_ini_file(section, key):
    config = configparser.ConfigParser()
    config.read('global.ini')
    return config.get(section, key)


if __name__ == '__main__':
    import warnings
    # 忽略所有警告
    warnings.filterwarnings("ignore")

    create_directory()
    params = Utils.get_init_params()
    Tp = float(__read_ini_file('network', 'Tp'))
    Tf = float(__read_ini_file('network', 'Tf'))
    wp = list(map(float, __read_ini_file('network', 'wp').split(',')))
    wf = list(map(float, __read_ini_file('network', 'wf').split(',')))
    evoCNN = evolveCNN(params)
