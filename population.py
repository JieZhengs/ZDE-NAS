import numpy as np


def initialize_population(params):
    pop_size = params['pop_size']
    max_length = params['max_length']
    max_output_channel = params['max_output_channel']
    population = []
    for _ in range(pop_size):
        num_net = int(max_length)

        part2 = []
        p_valid = np.random.random()
        for i in range(num_net):
            p_ = np.random.random()
            if p_ <= p_valid:
                num_feature_maps = np.random.randint(0, max_output_channel)
            else:
                p1 = np.random.random()
                if p1 <= 0.5:
                    num_feature_maps = np.random.randint(-30, 0)
                else:
                    num_feature_maps = np.random.randint(max_output_channel, max_output_channel + 30)
            part2.append(num_feature_maps)

        # num_stride2 = np.random.randint(0, max_stride2 + 1)
        # num_stride1 = num_net - num_stride2

        # find the position where the pooling layer can be connected
        # availabel_positions = list(idx for idx in range(0, num_net) if 0 <= part2[
        #     idx] < max_output_channel)  # only consider those valid layers as possible strided layers
        # if len(availabel_positions) == 0:
        #     part2[0] = np.random.randint(0, max_output_channel)
        #     availabel_positions = [0]
        # np.random.shuffle(availabel_positions)
        # np.random.shuffle(availabel_positions)
        # while len(availabel_positions) < num_stride2:
        #     supp_list = [idx for idx in range(0, num_net) if idx not in availabel_positions]
        #     availabel_positions.append(np.random.choice(supp_list, 1)[0])
        # select_positions = np.sort(availabel_positions[0:num_stride2])  # the positions of pooling layers in the net
        part1 = []
        # for i in range(num_net):
        #     if i in select_positions:
        #         code_stride2 = np.random.randint(3, 6)
        #         part1.append(code_stride2)
        #     else:
        #         code_stride1 = np.random.randint(0, 3)
        #         part1.append(code_stride1)
        for i in range(num_net):
            code_stride1 = np.random.randint(0, 8)
            part1.append(code_stride1)
        population.append([part1, part2])
    return population


def test_population():
    params = {}
    params['pop_size'] = 30
    params['max_length'] = 20
    params['image_channel'] = 3
    params['max_output_channel'] = 256
    pop = initialize_population(params)
    return pop


def get_valid_indi(individual):
    valid_part1 = []
    valid_part2 = []
    for i, element in enumerate(individual[1]):
        if 0 <= element <= 256 - 1:
            # valid_part1为卷积配置信息 valid_part2为输入输出通道信息
            valid_part1.append(individual[0][i])
            valid_part2.append(element)
    if len(valid_part1) == 0 and len(valid_part2) == 0:
        print('无效值')
        valid_part1.append(np.random.randint(0, 8))
        valid_part2.append(np.random.randint(0, 256))
    return [valid_part1, valid_part2]

def print_valid(population):
    for i, individual in enumerate(population):
        print(i, get_valid_indi(individual))

if __name__ == '__main__':
    p = test_population()
    print(p)
    print_valid(p)
