import os
import torch
import logging
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from util import constants as C


class TFLogger:
    def log_images(self, images, tag, size=125):
        """
        Log images and optionally detection to tensorboard
        :param logger: [Tensorboard Logger] Tensorboard logger object.
        :param images: [tensor] batch of images indexed
                    [batch, channel, size1, size2]
        TODO: Include an argument for image labels;
            Print the labels on the images.
        """
        images = prep_images_for_logging(images,
                                         pretrained=self.hparams['pretrained'],
                                         size=size)
        self.logger.experiment.add_images(tag, images)


class ReportLogger:
    def report(self, preds, labels, regions, areas):
        plt_labels = self.hparams['labels']
        plt_labels = [string.lower().replace('deforestation', '')
                                   for string in plt_labels]  # First class too long for x ticks

        plt_save_dir = self.hparams['default_save_path']
        if not os.path.exists(plt_save_dir):
            os.makedirs(plt_save_dir)

        # regionwise analysis
        if self.hparams['regions']:
            for region in C.CONTINENTS:
                region_preds = preds[regions == region]
                region_labels = labels[regions == region]

                # 'Smoothing'
                region_preds = np.concatenate(
                    [region_preds, np.arange(len(plt_labels))], 0)
                region_labels = np.concatenate(
                    [region_labels, np.arange(len(plt_labels))], 0)
                
                region_report = classification_report(
                            region_labels,
                            region_preds,
                            target_names=plt_labels)
                logging.info(f'Regional Class Report for {region}')
                logging.info(region_report)
                region_report_path = os.path.join(plt_save_dir, f'class_table_{region}.txt')
                region_heatmap_path = os.path.join(plt_save_dir, f'conf_heatmap_{region}.jpg')
                with open(region_report_path, 'w+') as fp:
                    fp.write(region_report)

                hm = sns.heatmap(confusion_matrix(region_labels, region_preds, normalize='true'), 
                                 cbar=True,
                                 xticklabels=plt_labels, 
                                 yticklabels=plt_labels, 
                                 square=True)
                plt.title(f'Conf matrix for {region}')
                plt.savefig(region_heatmap_path, bbox_inches='tight')
                plt.close()

        if areas is not None:
            logging.info(f'Eval weighted by loss area: ')
            report =  classification_report(
                        labels,
                        preds,
                        target_names=plt_labels,
                        sample_weight=areas)
            logging.info(report)
            print('Eval weighted by loss area: ')
            print(report)
            
        logging.info(f'Class report for all data: ')       
        report =  classification_report(
                    labels,
                    preds,
                    target_names=plt_labels)
        logging.info(report)
        print('Class report for all data (not weighted by loss area):')
        print(report)
        
        hm = sns.heatmap(confusion_matrix(labels, preds, normalize='true'), 
                         cbar=True,
                         xticklabels=plt_labels, 
                         yticklabels=plt_labels, 
                         square=True)
        overall_heatmap_path = os.path.join(plt_save_dir, 'conf_heatmap_overall.jpg')
        plt.savefig(overall_heatmap_path, bbox_inches='tight')


def prep_images_for_logging(images, pretrained=False,
                            size=125):
    """
    Prepare images to be logged
    :param images: [tensor] batch of images indexed
                   [channel, size1, size2]
    :param mean: [list] mean values used to normalize images
    :param std: [list] standard deviation values used to normalize images
    :param size: [int] new size of the image to be rescaled
    :return: images that are reversely normalized
    """
    if pretrained:
        mean = C.IMAGENET_MEAN
        std = C.IMAGENET_STD
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]
    # images = normalize_inverse(images, mean, std)
    images = F.interpolate(images, size=size,
                           mode='bilinear', align_corners=True)
    return images


def normalize_inverse(images, mean=C.IMAGENET_MEAN, std=C.IMAGENET_STD):
    """
    Reverse Normalization of Pytorch Tensor
    :param images: [tensor] batch of images indexed
                   [batch, channel, size1, size2]
    :param mean: [list] mean values used to normalize images
    :param std: [list] standard deviation values used to normalize images
    :return: images that are reversely normalized
    """
    mean_inv = torch.FloatTensor(
        [-m/s for m, s in zip(mean, std)]).view(1, 3, 1, 1)
    std_inv = torch.FloatTensor([1/s for s in std]).view(1, 3, 1, 1)
    if torch.cuda.is_available():
        mean_inv = mean_inv.cuda()
        std_inv = std_inv.cuda()
    return (images - mean_inv) / std_inv
