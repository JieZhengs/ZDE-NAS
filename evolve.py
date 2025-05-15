import copy

import numpy as np

class FDE(object):
    def __init__(self, individual, idx, population, acc_set, params, _log):

        self.individual = individual
        self.idx = idx
        self.population = population
        self.acc_set = acc_set
        self.params = params
        self.diff_rate = params['diff_rate']
        self.crossover_rate = params['crossover_rate']
        self.max_output_channel = params['max_output_channel']
        self.log = _log

    def obtain_better_inds_idxs(self):
        acc_cur = self.acc_set[self.idx]
        idxs = []
        for idx in range(len(self.acc_set)):
            if idx != self.idx:
                if self.acc_set[idx] >= acc_cur:
                    idxs.append(idx)
        return idxs

    def mutate(self):
        # 从最优个体群里随机选择一个好的个体
        better_inds_idxs = self.obtain_better_inds_idxs()
        # print(better_inds_idxs)
        if better_inds_idxs:
            id0 = np.random.choice(better_inds_idxs, 1)[0]
            x0 = np.asarray(self.population[id0])
        else:
            x0 = np.asarray(self.individual)
        pop_size = len(self.population)

        # 从其他群体里随机选择两个个体
        idxs = np.random.choice(list(range(pop_size)), 4, replace=False)
        while self.idx != None and self.idx in idxs:
            # print('在这里面')
            idxs = np.random.choice(list(range(pop_size)), 4, replace=False)
        x1 = np.asarray(self.population[idxs[0]])
        x2 = np.asarray(self.population[idxs[1]])
        x3 = np.asarray(self.population[idxs[2]])
        x4 = np.asarray(self.population[idxs[3]])

        print('开始交叉变异操作')
        # x0为其中一个好的个体  x1和x2为随机得到的个体
        #part1 mutation 离散的形式
        indi_mut_1 = []
        # 从x1中删除在x2中共同出现的元素
        diff_v1 = [x1[0][i] if not x1[0][i]==x2[0][i] else 0 for i in range(len(x1[0]))]
        diff_v2 = [x3[0][i] if not x3[0][i]==x4[0][i] else 0 for i in range(len(x3[0]))]
        diff_part1 = [diff_v1[i] if not diff_v1[i]==diff_v2[i] else 0 for i in range(len(diff_v1))]
        for i in range(len(x0[0])):
            p_ = np.random.random()
            # 进行突变
            if p_ <= self.diff_rate:
                if diff_part1[i] == 0:
                    rand_opt = np.random.randint(0, 8)
                    indi_mut_1.append(rand_opt)
                else:
                    indi_mut_1.append(diff_part1[i])
            # 不进行突变
            else:
                indi_mut_1.append(x0[0][i])

        #part2 mutation 连续的形式
        # indi_mut_2 = np.asarray(self.individual[1]) + self.diff_rate*(x0[1] - np.asarray(self.individual[1])) + self.diff_rate*(x1[1] - x2[1])
        # indi_mut_2 = list(map(int, indi_mut_2))

        # part2 mutation 离散的形式
        indi_mut_2 = []
        diff_v3 = [x1[1][i] if not x1[1][i]==x2[1][i] else 0 for i in range(len(x1[1]))]
        diff_v4 = [x3[1][i] if not x3[1][i]==x4[1][i] else 0 for i in range(len(x3[1]))]
        diff_part2 = [diff_v3[i] if not diff_v3[i]==diff_v4[i] else 0 for i in range(len(diff_v3))]

        for i in range(len(x0[0])):
            p_ = np.random.random()
            # 进行突变
            if p_ <= self.diff_rate:
                if diff_part2[i] == 0:
                    rand_filters = np.random.randint(0, self.max_output_channel)
                    indi_mut_2.append(rand_filters)
                else:
                    indi_mut_2.append(diff_part2[i])
            # 不进行突变
            else:
                indi_mut_2.append(x0[0][i])

        return [indi_mut_1, indi_mut_2]

    def crossover(self, indi_mut):
        offspring_part1 = []
        offspring_part2 = []
        [indi_mut_1, indi_mut_2] = indi_mut
        j = np.random.choice(len(self.individual[0]))
        for i in range(len(self.individual[0])):
            p_ = np.random.random()
            if p_ <= self.crossover_rate or i==j:
                offspring_part1.append(indi_mut_1[i])
                offspring_part2.append(indi_mut_2[i])
            else:
                offspring_part1.append(self.individual[0][i])
                offspring_part2.append(self.individual[1][i])
        indi_mut_ = self.adjust_Indi([offspring_part1, offspring_part2])
        return indi_mut_
    def get_valid_indi(self, individual):
        valid_part1 = []
        valid_part2 = []
        for i, element in enumerate(individual[1]):
            if 0<=element<self.params['max_output_channel']:
                valid_part1.append(individual[0][i])
                valid_part2.append(element)
        if len(valid_part1) == 0 and len(valid_part2) == 0:
            valid_part1.append(np.random.randint(0, 8))
            valid_part2.append(np.random.randint(0, self.params['max_output_channel']))
        return [valid_part1, valid_part2]

    def adjust_Indi(self, individual):
        valid_indi = self.get_valid_indi(individual)
        if len(valid_indi[0]) == 0:
            individual[0][0] = np.random.randint(0, 8)
            individual[1][0] = np.random.randint(0, self.params['max_output_channel'])
            return individual
        else:
            part2 = individual[1]
            individual = copy.deepcopy(individual)
            for i, element in enumerate(part2):
                if element > self.params['max_output_channel'] + 30:
                    individual[1][i] = self.params['max_output_channel'] + 30
                elif element < -30:
                    individual[1][i] = -30
            return individual

    def _count_stride2(self, individual):
        pos_stride2 = [0]*(len(individual[0]))
        for i, element in enumerate(individual[0]):
            if element >= 3:
                pos_stride2[i] = 1
        return pos_stride2




