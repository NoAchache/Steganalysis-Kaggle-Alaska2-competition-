# https://github.com/DXQer/Cov-Pooling-Steganalytic-Network/blob/master/srm_filter_kernel.py

import numpy as np
import torch

class SrmFiltersSetter:

    @staticmethod
    def get_filters():
        filter_class_1 = [
            np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 1],
                [0, -1, 0],
                [0, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0],
                [1, -1, 0],
                [0, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0],
                [0, -1, 1],
                [0, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0],
                [0, -1, 0],
                [1, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0],
                [0, -1, 0],
                [0, 1, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ], dtype=np.float32)
        ]

        filter_class_2 = [
            np.array([
                [1, 0, 0],
                [0, -2, 0],
                [0, 0, 1]
            ], dtype=np.float32),
            np.array([
                [0, 1, 0],
                [0, -2, 0],
                [0, 1, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 1],
                [0, -2, 0],
                [1, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0],
                [1, -2, 1],
                [0, 0, 0]
            ], dtype=np.float32),
        ]

        filter_class_3 = [
            np.array([
                [-1, 0, 0, 0, 0],
                [0, 3, 0, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, -1, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0, 0, -1],
                [0, 0, 0, 3, 0],
                [0, 0, -3, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -3, 3, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 0, 3, 0],
                [0, 0, 0, 0, -1]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, -1, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, -3, 0, 0],
                [0, 3, 0, 0, 0],
                [-1, 0, 0, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 3, -3, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=np.float32)
        ]

        filter_edge_3x3 = [
            np.array([
                [-1, 2, -1],
                [2, -4, 2],
                [0, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 2, -1],
                [0, -4, 2],
                [0, 2, -1]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0],
                [2, -4, 2],
                [-1, 2, -1]
            ], dtype=np.float32),
            np.array([
                [-1, 2, 0],
                [2, -4, 0],
                [-1, 2, 0]
            ], dtype=np.float32),
        ]

        filter_edge_5x5 = [
            np.array([
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=np.float32),
            np.array([
                [0, 0, -2, 2, -1],
                [0, 0, 8, -6, 2],
                [0, 0, -12, 8, -2],
                [0, 0, 8, -6, 2],
                [0, 0, -2, 2, -1]
            ], dtype=np.float32),
            np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]
            ], dtype=np.float32),
            np.array([
                [-1, 2, -2, 0, 0],
                [2, -6, 8, 0, 0],
                [-2, 8, -12, 0, 0],
                [2, -6, 8, 0, 0],
                [-1, 2, -2, 0, 0]
            ], dtype=np.float32),
        ]

        square_3x3 = np.array([
            [-1, 2, -1],
            [2, -4, 2],
            [-1, 2, -1]
        ], dtype=np.float32)

        square_5x5 = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=np.float32)

        filter_class_1 = [np.array([hpf] * 3) for hpf in filter_class_1]
        normalized_filter_class_2 = [np.array([hpf / 2]*3) for hpf in filter_class_2]
        normalized_filter_class_3 = [np.array([hpf / 3]*3) for hpf in filter_class_3]
        normalized_filter_edge_3x3 = [np.array([hpf / 4]*3) for hpf in filter_edge_3x3]
        normalized_square_3x3 = np.array([square_3x3 / 4]*3)
        normalized_filter_edge_5x5 = [np.array([hpf / 12]*3) for hpf in filter_edge_5x5]
        normalized_square_5x5 = np.array([square_5x5 / 12]*3)

        normalized_hpf_3x3_list = filter_class_1 + normalized_filter_class_2 + normalized_filter_edge_3x3 + [
            normalized_square_3x3]
        normalized_hpf_5x5_list = normalized_filter_class_3 + normalized_filter_edge_5x5 + [normalized_square_5x5]
        return np.array(normalized_hpf_3x3_list), np.array(normalized_hpf_5x5_list)

    @staticmethod
    def initialize_filters(model):
        normalized_hpf_3x3_list, normalized_hpf_5x5_list = SrmFiltersSetter.get_filters()
        model.srm.conv3x3.weight = torch.nn.Parameter(torch.tensor(normalized_hpf_3x3_list))
        model.srm.conv5x5.weight = torch.nn.Parameter(torch.tensor(normalized_hpf_5x5_list))
