"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from . import craft_utils
from . import imgproc
from . import file_utils
import json
import zipfile

from .craft import CRAFT

from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

class InferCRAFT:
    def __init__(self, trained_model='weights/craft_mlt_25k.pth', text_threshold=0.7, low_text=0.4, link_threshold=0.4,
                 cuda=True, canvas_size=1280, mag_ratio=1.5, poly=False, show_time=False, test_folder='/data/',
                 refine=False, refiner_model='weights/craft_refiner_CTW1500.pth', result_folder='/detection_results/'):
        self.trained_model = trained_model
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.cuda = cuda
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.poly = poly
        self.show_time = show_time
        self.test_folder = test_folder
        self.refine = refine
        self.refiner_model = refiner_model
        self.result_folder = result_folder

        # For test images in a folder
        self.image_list, _, _ = file_utils.get_files(self.test_folder)

        if not os.path.isdir(self.result_folder):
            os.mkdir(self.result_folder)

    def test_net(self, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
        t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.canvas_size,
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if self.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        return boxes, polys, ret_score_text
    
    def run(self):
        # load net
        net = CRAFT()     # initialize

        print('Loading weights from checkpoint (' + self.trained_model + ')')
        if self.cuda:
            net.load_state_dict(copyStateDict(torch.load(self.trained_model)))
        else:
            net.load_state_dict(copyStateDict(torch.load(self.trained_model, map_location='cpu')))

        if self.cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

        net.eval()

        # LinkRefiner
        refine_net = None
        if self.refine:
            from refinenet import RefineNet
            refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + self.refiner_model + ')')
            if self.cuda:
                refine_net.load_state_dict(copyStateDict(torch.load(self.refiner_model)))
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(copyStateDict(torch.load(self.refiner_model, map_location='cpu')))

            refine_net.eval()
            self.poly = True

        t = time.time()

        # load data
        for k, image_path in enumerate(self.image_list):
            print("Test image {:d}/{:d}: {:s}".format(k+1, len(self.image_list), image_path), end='\r')
            image = imgproc.loadImage(image_path)

            bboxes, polys, score_text = self.test_net(net, image, self.text_threshold, self.link_threshold, self.low_text, self.cuda, self.poly, refine_net)

            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            mask_file = self.result_folder + "/res_" + filename + '_mask.jpg'
            cv2.imwrite(mask_file, score_text)

            file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=self.result_folder)

        print("elapsed time : {}s".format(time.time() - t))