import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn
import os.path

from models import get_model
from eval import get_loss_fn, get_accuracy, AverageMeter
from data import HansenDriversDataset
import util.constants as C
from .logger import TFLogger, ReportLogger


class Model(pl.LightningModule, TFLogger, ReportLogger):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super(Model, self).__init__()
        self.save_hyperparameters(params)
        self.model = get_model(self.hparams)
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()
        self.val_loss = AverageMeter()
        self.val_acc = AverageMeter()

        if self.hparams['output_pre_logits']:
            self.pool = nn.AdaptiveAvgPool2d(1)

        if self.hparams["class_weight"]:
            class_weights = torch.Tensor(
                self.get_dataset(C.TRAIN_SPLIT).class_weights())
            class_weights = class_weights / class_weights.sum()
            self.loss = get_loss_fn(self.hparams, class_weights=class_weights)
        else:
            self.loss = get_loss_fn(self.hparams)

    def _extract_pre_logits(self, batch):
        if hasattr(self.model, 'extract_pre_logits'):
            x = self.model.extract_pre_logits(batch)
        elif hasattr(self.model, 'get_logits'):
            x = self.model.get_logits(batch)
        elif hasattr(self.model, 'extract_features'):
            x = self.model.extract_features(batch['image'])
        else:
            raise ValueError(
                f"Model {type(self.model).__name__} does not expose a "
                "feature extractor for output_pre_logits."
            )

        if x.ndim > 2:
            x = self.pool(x).view(x.size(0), -1)
        return x

    def forward(self, batch):
        y = batch['label']
        logits = self.model(batch)
        loss = self.loss(logits, y)
        acc = get_accuracy(logits, y)
        return logits, loss, acc


    def on_epoch_start(self):
        self.train_loss.reset()
        self.train_acc.reset()
        self.val_loss.reset()
        self.val_acc.reset()

    def training_step(self, batch, batch_nb):
        """
        Returns:
            A dictionary of loss and metrics, with:
                loss(required): loss used to calculate the gradient
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        logits, loss, acc = self.forward(batch)
        n = self.hparams['batch_size']

        self.train_loss.update(loss.item(), n)
        self.train_acc.update(acc, n)

        logs = {
            'loss': loss,
            'train_acc': acc,
            'log': {'train_loss': loss,
                    'train_acc': acc,
                    'avg_train_loss': self.train_loss.get_average(),
                    'avg_train_acc': self.train_acc.get_average(),
                    'epoch': self.current_epoch},
            'progress_bar': {'avg_train_loss': self.train_loss.get_average(),
                             'avg_train_acc': self.train_acc.get_average()}
        }
        return logs

    def validation_step(self, batch, batch_nb):
        logits, loss, acc = self.forward(batch)
        self.val_loss.update(loss.item())
        self.val_acc.update(acc)

        logs = {
            'val_loss': loss,
            'val_acc': acc,
            'log': {'val_loss': loss,
                    'val_acc': acc,
                    'epoch': self.current_epoch},
            'progress_bar': {'val_loss': loss}
        }
        return logs

    def validation_epoch_end(self, outputs):
        """
        Aggregate and return the validation metrics

        Args:
        outputs: A list of dictionaries of metrics from `validation_step()'
        Returns: None
        Returns:
            A dictionary of loss and metrics, with:
                val_loss (required): validation_loss
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        avg_loss = torch.stack([out['val_loss'] for out in outputs]).mean()
        avg_acc = torch.stack([out['val_acc'] for out in outputs]).mean()
        metrics = {
            'avg_val_loss': avg_loss,
            'avg_val_acc': avg_acc
        }

        logs = {
            'val_loss': avg_loss,
            'log': metrics,
            'progress_bar': metrics
        }
        return logs

    def test_step(self, batch, batch_nb):
        if self.hparams['output_pre_logits']:
            x = self._extract_pre_logits(batch)
        label, index, event_index, gooder_id = batch['label'], batch['index'], batch['event_index'], batch['gooder_id']
        
        logits, loss, acc = self.forward(batch)
        pred_label = torch.argmax(logits, -1).cpu().numpy()
        probs = torch.nn.functional.softmax(logits, -1)
        prob = torch.max(probs, -1).values

        logs = {
            'test_loss': loss,
            'test_acc': acc,
            'log': {'test_loss': loss,
                    'test_acc': acc},
            'labels': label.cpu().numpy(),
            'pred_labels': pred_label,
            'prob': prob.cpu().numpy(),
            'index': index.cpu().numpy(),
            'event_index': event_index.cpu().numpy(),
            'gooder_id': gooder_id
        }
        if self.hparams['output_logits']:
            logs['logits'] = logits
        if self.hparams['output_pre_logits']:
            logs['pre_logits'] = x
        
        if 'region' in batch:
            logs['region'] = batch['region']
        if self.hparams['eval_by_pixel']:
            logs['area'] = batch['loss_areas']
            
        return logs

    def test_epoch_end(self, outputs):
        if self.hparams['output_logits']:
            logits = torch.cat([out['logits'] for out in outputs])
            path = os.path.join(self.hparams['save_dir'], self.hparams['exp_name'], "logits_" + self.hparams['test_split'] + ".csv")
            gooder_ids = torch.cat([out['gooder_id'] for out in outputs])
            gooder_ids = pd.DataFrame(gooder_ids.cpu().numpy())
            gooder_ids.columns = ['GoodeR_ID']
            logits = pd.DataFrame(logits.cpu().numpy())
            data = pd.concat([gooder_ids, logits], axis=1)
            data.to_csv(path, index=False)
        if self.hparams['output_pre_logits']:
            pre_logits = torch.cat([out['pre_logits'] for out in outputs])
            path = os.path.join(self.hparams['save_dir'], self.hparams['exp_name'], "pre_logits_" + self.hparams['test_split'] + ".csv")
            gooder_ids = torch.cat([out['gooder_id'] for out in outputs])
            gooder_ids = pd.DataFrame(gooder_ids.cpu().numpy())
            gooder_ids.columns = ['GoodeR_ID']
            pre_logits = pd.DataFrame(pre_logits.cpu().numpy())
            data = pd.concat([gooder_ids, pre_logits], axis=1)
            data.to_csv(path, index=False)

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        labels = np.stack([x['labels'] for x in outputs])
        preds = np.stack([x['pred_labels'] for x in outputs])
        probs = np.stack([x['prob'] for x in outputs])
        indices = np.stack([x['index'] for x in outputs])
        event_indices = np.stack([x['event_index'] for x in outputs])
        regions = None
        if outputs and 'region' in outputs[0]:
            regions = np.stack([x['region'] for x in outputs])
        
        if self.hparams['eval_by_pixel']:
            areas = np.stack([x['area'].cpu() for x in outputs])
        else:
            areas = None
        
        self.report(preds, labels, regions, areas)

        correct_images = []
        incorrect_images = []

        test_dataset = self.get_dataset(self.hparams['test_split'])
        save_labels = [label.replace(' ', '_') for label in C.HANSEN_LABELS_V3]

        for i in tqdm(range(len(indices)), desc="Logging images to TensorBoard"):
            index = indices[i][0]
            event_index = event_indices[i][0]
            image = C.TOTENSOR_TRANSFORM(test_dataset._get_image(index)[0])
            if labels[i] == preds[i]:
                tag = 'correct'
            else:
                tag = 'incorrect'
            label = save_labels[int(labels[i][0])]
            pred = save_labels[int(preds[i][0])]
            tag += f'_label_{label}_pred_{pred}_event_index_{event_index}_prob_{probs[i][0]:.2f}'
            images = torch.stack([image])
            self.log_images(images, tag)

        return {
            'avg_test_loss': avg_loss,
            'avg_test_acc': avg_acc
        }
   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'])
        scheduler = {
            # Tracking val acc so mode max
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=4, mode='max'),
            'monitor': 'avg_val_acc',  # Default: val_loss
            'interval': 'epoch',
            'frequency': 1
            }
        if self.hparams['lr_schedule']:
            return [optimizer], [scheduler]
        return [optimizer]

    def train_dataloader(self):
        dataset = self.get_dataset(C.TRAIN_SPLIT)
        dl = DataLoader(dataset,
                        batch_size=self.hparams['batch_size'],
                        num_workers=self.hparams['num_dl_workers'],
                        shuffle=True)
        return dl

    def val_dataloader(self):
        dataset = self.get_dataset(C.VAL_SPLIT)
        dl = DataLoader(dataset,
                        batch_size=1,
                        num_workers=self.hparams['num_dl_workers'],
                        shuffle=False)
        return dl

    def test_dataloader(self):
        dataset = self.get_dataset(self.hparams['test_split'])
        dl = DataLoader(dataset,
                        batch_size=1,
                        num_workers=self.hparams['num_dl_workers'],
                        shuffle=False)
        return dl

    def get_dataset(self, split):
        transforms = self.get_transforms_list(split)
        if split == C.TRAIN_SPLIT:
            img_option = self.hparams['train_img_option']
        else:
            img_option = self.hparams['eval_img_option']
        dataset = HansenDriversDataset(
            self.hparams['dataset'],
            data_split=split,
            transforms=transforms,
            regions=self.hparams['regions'],
            img_option=img_option,
            first_last=self.hparams['first_last'],
            lrcn=self.hparams['lrcn'],
            load_polygon_loss=self.hparams['load_polygon_loss'],
            late_fusion_regions=self.hparams['late_fusion_regions'],
            load_aux=self.hparams['load_aux'],
            load_mode=self.hparams['load_mode'],
            year_cutoff=self.hparams['year_cutoff']
        )
        return dataset

    def get_transforms_list(self, split):

        zoomed = self.hparams['zoomed_regions']
        augmentation = self.hparams['augmentation']
        if split != C.TRAIN_SPLIT:
            augmentation = "none"

        resize_crop_transform = C.RESIZE_CROP_TRANSFORM[zoomed][split]
        augmentation_transform = C.AUGMENTATION_TRANSFORM[augmentation]
        totensor_transform = C.TOTENSOR_TRANSFORM
        transforms = [resize_crop_transform,
                      augmentation_transform,
                      totensor_transform]
        if self.hparams['image_net_norm']:
            transforms.append(C.IMAGE_NET_TRANSFORMS) 

        return transforms
