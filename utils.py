import configparser
import copy
import logging
import time
import sys
import os
from subprocess import Popen, PIPE

import numpy as np

strides = [1, 1, 1, 1, 1, 2, 2, 2]
start_dw_kernel_sizes = [0, 0, 3, 3, 3, 3, 5, 5]
middle_dw_kernel_sizes = [0, 0, 3, 3, 3, 3, 5, 5]
middle_dw_downsamples = [0, 0, 1, 1, 1, 1, 1, 1]
expand_ratios = [2, 2, 3, 3, 4, 4, 6, 6]
attentions = [0, 0, 0, 1, 1, 1, 1, 1]


class Utils(object):
    @classmethod
    def get_init_params(cls):
        params = {}
        params['pop_size'] = cls.get_params('settings', 'pop_size')
        params['num_iteration'] = cls.get_params('settings', 'num_iteration')
        params['diff_rate'] = float(cls.__read_ini_file('settings', 'diff_rate'))
        params['crossover_rate'] = float(cls.__read_ini_file('settings', 'crossover_rate'))
        params['max_length'] = cls.get_params('network', 'max_length')
        params['image_channel'] = cls.get_params('network', 'image_channel')
        params['max_output_channel'] = cls.get_params('network', 'max_output_channel')
        params['num_class'] = cls.get_params('network', 'num_class')
        params['epoch_test'] = cls.get_params('network', 'epoch_test')

        return params

    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read('global.ini')
        return config.get(section, key)

    @classmethod
    def get_params(cls, domain, key):
        rs = cls.__read_ini_file(domain, key)
        return int(rs)

    @classmethod
    def save_population_and_acc(cls, type, population, acc_set, num_parameters, flops, gen_no):
        """
        将种群decode之后，将其与eval的acc一起保存
        :param type: 类型，字符串，{'population', ‘pbest’, 'gbest'}
        :param population: list，种群，由各particle编码组成
        :param acc_set: eval获得的种群各particle的准确率
        :param gen_no: 种群的代数
        :return:
        """
        file_name = './populations/' + type + '_%02d.txt' % (gen_no)
        _str = cls.popAndAcc2str(population, acc_set, num_parameters, flops)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()

    @classmethod
    def get_valid_indi(cls, individual):
        valid_part1 = []
        valid_part2 = []
        for i, element in enumerate(individual[1]):
            if 0 <= element <= cls.get_params('network', 'max_output_channel') - 1:
                # valid_part1为卷积配置信息 valid_part2为输入输出通道信息
                valid_part1.append(individual[0][i])
                valid_part2.append(element)
        if len(valid_part1) == 0 and len(valid_part2) == 0:
            valid_part1.append(np.random.randint(0, 8))
            valid_part2.append(np.random.randint(0, cls.get_params('network', 'max_output_channel')))
        return [valid_part1, valid_part2]

    @classmethod
    def popAndAcc2str(cls, population, acc_set, num_parameters, flops):
        pop_str = []
        for id, individual in enumerate(population):
            _str = []
            _str.append('indi:%02d' % (id))
            _str.append('particle:%s' % (','.join(list(map(str, individual)))))
            _str.append('num_parameters:%d' % (num_parameters[id]))
            _str.append('flops:%.4f' % (flops[id]))
            _str.append('zen_score:%.4f' % (acc_set[id]))
            valid_indi = cls.get_valid_indi(individual)
            for i in range(len(valid_indi[0])):
                _sub_str = []
                _sub_str.append('layer:%02d' % (i))
                start_dw_kernel_size = start_dw_kernel_sizes[valid_indi[0][i]]
                middle_dw_kernel_size = middle_dw_kernel_sizes[valid_indi[0][i]]
                middle_dw_downsample = middle_dw_downsamples[valid_indi[0][i]]
                stride = strides[valid_indi[0][i]]
                expand_ratio = expand_ratios[valid_indi[0][i]]
                attention = attentions[valid_indi[0][i]]
                oup = valid_indi[1][i] + 1

                if i == 0:
                    in_channel = cls.get_params('network', 'image_channel')
                else:
                    in_channel = pre_out_channel

                _sub_str.append('UniversalInvertedBottleneckBlock')
                _sub_str.append('start_dw_kernel_size:%d' % (start_dw_kernel_size))
                _sub_str.append('middle_dw_kernel_size:%d' % (middle_dw_kernel_size))
                _sub_str.append('middle_dw_downsample:%d' % (middle_dw_downsample))
                _sub_str.append('stride:%d' % (stride))
                _sub_str.append('expand_ratio:%d' % (expand_ratio))
                _sub_str.append('attention:%d' % (attention))
                _sub_str.append('in_channel:%d' % (in_channel))
                _sub_str.append('out_channel:%d' % (oup))

                pre_out_channel = oup

                _str.append('%s%s%s' % ('[', ','.join(_sub_str), ']'))
            individual_str = '\n'.join(_str)
            pop_str.append(individual_str)
            pop_str.append('-' * 100)
        return '\n'.join(pop_str)

    @classmethod
    def read_template(cls):
        dataset = str(cls.__read_ini_file('settings', 'dataset'))
        _path = './template/' + dataset + '.py'
        part1 = []
        part2 = []
        part3 = []

        f = open(_path)
        f.readline()  # skip this comment
        line = f.readline().rstrip()
        while line.strip() != '#generated_init':
            part1.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part1))

        line = f.readline().rstrip()  # skip the comment '#generated_init'
        while line.strip() != '#generate_forward':
            part2.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part2))

        line = f.readline().rstrip()  # skip the comment '#generate_forward'
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part3))
        return part1, part2, part3

    @classmethod
    def generate_pytorch_file(cls, individual, curr_gen, id):
        # query convolution unit
        conv_list = []
        valid_indi = cls.get_valid_indi(individual)
        for i in range(len(valid_indi[0])):
            start_dw_kernel_size = start_dw_kernel_sizes[valid_indi[0][i]]
            middle_dw_kernel_size = middle_dw_kernel_sizes[valid_indi[0][i]]
            middle_dw_downsample = middle_dw_downsamples[valid_indi[0][i]]
            stride = strides[valid_indi[0][i]]
            expand_ratio = expand_ratios[valid_indi[0][i]]
            attention = attentions[valid_indi[0][i]]
            oup = valid_indi[1][i] + 1

            conv_name = 'self.conv_%d' % (i)

            if i == 0:
                in_channel = cls.get_params('network', 'image_channel')
            else:
                in_channel = pre_out_channel

            conv = '%s = UniversalInvertedBottleneckBlock(inp=%d, oup=%d, start_dw_kernel_size=%d, middle_dw_kernel_size=%d, middle_dw_downsample=%d, stride=%d, expand_ratio=%d, attention=%d)' % (
                conv_name, in_channel, oup, start_dw_kernel_size, middle_dw_kernel_size, middle_dw_downsample, stride,
                expand_ratio, attention)
            conv_list.append(conv)
            pre_out_channel = oup

        # query fully-connect layer, cause a global avg_pooling layer is added before the fc layer, so the input size
        # of the fc layer is equal to out channel
        conv_end1 = 'self.conv_end1 = nn.Conv2d(%d, %d, kernel_size=1, stride=1, bias=False)' % (
            pre_out_channel, pre_out_channel)
        conv_list.append(conv_end1)
        fully_layer_name1 = 'self.linear = nn.Linear(%d, %d)' % (
            pre_out_channel, cls.get_params('network', 'num_class'))

        # generate the forward part
        forward_list = []
        for i in range(len(valid_indi[0])):
            if i == 0:
                last_out_put = 'x'
            else:
                last_out_put = 'out_%d' % (i - 1)
            _str = 'out_%d = self.conv_%d(%s)' % (i, i, last_out_put)
            forward_list.append(_str)

        forward_list.append('out = out_%d' % (len(valid_indi[0]) - 1))
        # forward_list.append('out = F.adaptive_avg_pool2d(out,(1,1))')

        part1, part2, part3 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\n        %s' % ('#conv unit'))
        for s in conv_list:
            _str.append('        %s' % (s))
        _str.append('\n        %s' % ('#linear unit'))
        _str.append('        %s' % (fully_layer_name1))
        # _str.append('        %s' % (fully_layer_name2))

        _str.extend(part2)
        for s in forward_list:
            _str.append('        %s' % (s))
        _str.extend(part3)
        # print('\n'.join(_str))
        file_path = './scripts/indi%02d_%02d_Zen_Score.py' % (curr_gen, id)
        script_file_handler = open(file_path, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()
        file_name = 'indi%02d_%02d_Zen_Score' % (curr_gen, id)
        return file_name


class Log(object):
    _logger = None

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("EDE-NAS")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)

    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warning(_str)


class GPUTools(object):
    @classmethod
    def _get_equipped_gpu_ids_and_used_gpu_info(cls):
        p = Popen('nvidia-smi', stdout=PIPE)
        output_info = p.stdout.read().decode('UTF-8')
        lines = output_info.split(os.linesep)
        equipped_gpu_ids = []
        for line_info in lines:
            if not line_info.startswith(' '):
                if 'GeForce' in line_info or 'Quadro' in line_info or 'Tesla' in line_info or 'RTX' in line_info or 'A40' in line_info:
                    equipped_gpu_ids.append(line_info.strip().split(' ', 4)[3])
            else:
                break

        gpu_info_list = []
        for line_no in range(len(lines) - 3, -1, -1):
            if lines[line_no].startswith('|==='):
                break
            else:
                gpu_info_list.append(lines[line_no][1:-1].strip())

        return equipped_gpu_ids, gpu_info_list

    @classmethod
    def get_available_gpu_ids(cls):
        equipped_gpu_ids, gpu_info_list = cls._get_equipped_gpu_ids_and_used_gpu_info()

        used_gpu_ids = []

        for each_used_info in gpu_info_list:
            if 'python' in each_used_info:
                used_gpu_ids.append((each_used_info.strip().split(' ', 1)[0]))

        unused_gpu_ids = []
        for id_ in equipped_gpu_ids:
            if id_ not in used_gpu_ids:
                unused_gpu_ids.append(id_)
        return unused_gpu_ids

    @classmethod
    def detect_available_gpu_id(cls):
        unused_gpu_ids = cls.get_available_gpu_ids()
        if len(unused_gpu_ids) == 0:
            Log.info('GPU_QUERY-No available GPU')
            return None
        else:
            Log.info('GPU_QUERY-Available GPUs are: [%s], choose GPU#%s to use' % (
                ','.join(unused_gpu_ids), unused_gpu_ids[0]))
            return int(unused_gpu_ids[0])

    @classmethod
    def all_gpu_available(cls):
        _, gpu_info_list = cls._get_equipped_gpu_ids_and_used_gpu_info()

        used_gpu_ids = []

        for each_used_info in gpu_info_list:
            if 'python' in each_used_info:
                used_gpu_ids.append((each_used_info.strip().split(' ', 1)[0]))
        if len(used_gpu_ids) == 0:
            Log.info('GPU_QUERY-None of the GPU is occupied')
            return True
        else:
            Log.info('GPU_QUERY- GPUs [%s] are occupying' % (','.join(used_gpu_ids)))
            return False
