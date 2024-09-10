# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-02 14:38:36
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-03 09:23:07
# @Email:  cshzxie@gmail.com

import numpy as np
import torch
import transforms3d

class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)
            if transform.__class__ in [NormalizeObjectPose]:
                data = transform(data)
            else:
                for k, v in data.items():
                    if k in objects and k in data:
                        if transform.__class__ in [
                            RandomMirrorPoints
                        ]:
                            data[k] = transform(v, rnd_value)
                        else:
                            data[k] = transform(v)

        return data

class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud

class UpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(self.n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud
class UpSamplePoints1(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            # 如果点数多于所需，随机选择 self.n_points 个点
            return ptcloud[np.random.permutation(curr)[:self.n_points]]

        while curr < self.n_points:
            # 点云加倍直到至少达到所需点数
            ptcloud = np.tile(ptcloud, (2, 1))
            curr = ptcloud.shape[0]

        if curr > self.n_points:
            # 如果点数超过所需，随机选择 self.n_points 个点
            ptcloud = ptcloud[np.random.permutation(curr)[:self.n_points]]

        return ptcloud
class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value > 0.5 and rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]

        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data
    
class AddNoiseToPointCloud(object):
    def __init__(self, parameters):
        self.noise_level=parameters['noise_level']

    def __call__(self, ptcloud):
        # 添加噪声
        # mean = np.mean(np.array(ptcloud[:, :3]), axis=0)
        std_devs = np.std(np.array(ptcloud[:, :3]), axis=0)
        # print(mean,std_devs)
        noise = np.random.normal(scale=self.noise_level, size=ptcloud[:, :3].shape) * std_devs
        # noise = np.random.normal(scale=self.noise_level, size=ptcloud[:, :2].shape)
        ptcloud[:, :3] += noise

        return ptcloud
    
class RotatePointsClockwise90(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud):
        # 创建一个旋转矩阵，将点云顺时针旋转 90 度
        angle = np.pi / 2  # 90 度对应的弧度
        rotation_matrix = transforms3d.axangles.axangle2mat([0, 0, -1], angle)

        # 对点云进行旋转变换
        ptcloud[:, :3] = np.dot(ptcloud[:, :3], rotation_matrix.T)

        return ptcloud
    
class RotatePointsClockwise180(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud):
        # 创建一个旋转矩阵，将点云顺时针旋转180度
        angle = np.pi  # 180度对应的弧度
        rotation_matrix = transforms3d.axangles.axangle2mat([0, 0, -1], angle)

        # 对点云进行旋转变换
        ptcloud[:, :3] = np.dot(ptcloud[:, :3], rotation_matrix.T)

        return ptcloud
    
class ScalePointCloud(object):
    def __init__(self, parameters):
        self.scale_factor=parameters['scale_factor']

    def __call__(self, ptcloud):
        # 创建一个缩放矩阵
        scale_matrix = np.eye(3) * self.scale_factor

        # 对点云进行缩放变换
        ptcloud[:, :3] = np.dot(ptcloud[:, :3], scale_matrix.T)

        return ptcloud
    
class TranslatePointCloud(object):
    def __init__(self, parameters):
        self.translation_vector=parameters['translation_vector']

    def __call__(self, ptcloud):
        # 对点云进行偏移变换
        ptcloud[:, :3] += self.translation_vector

        return ptcloud