import os
import fire
import logging
import torch
from pytorch_lightning import Trainer
from pathlib import Path

from lightning.model import Model
from lightning.util import (get_ckpt_callback,
                            get_early_stop_callback,
                            get_logger)
from models import train_baseline
from util import init_exp_folder, Args
from util import constants as C 
from datetime import datetime

def train(exp_name=None,
          model="DenseNet121",
          dataset=str(C.HANSEN_V5_DIR),
          lr=3e-5,
          weight_decay=1e-2,
          batch_size=16,
          gpus=None,
          num_dl_workers=8,
          pretrained=False,
          num_classes=len(C.HANSEN_LABELS_V3),
          labels=C.HANSEN_LABELS_V3,
          log_save_interval=1,
          distributed_backend="dp",
          gradient_clip_val=0.5,
          max_epochs=100,
          patience=10,
          train_percent_check=1.0,
          save_dir=str(C.SANDBOX_DIR),
          tb_path=str(C.TB_DIR),
          loss_fn="CE",
          weights_summary=None,
          zoomed_regions=False,
          augmentation="none",
          image_net_norm=False,
          lr_schedule=True,
          ckpt_path=None,
          class_weight=False,
          regions=None,
          composite=True,
          train_img_option=C.IMG_OPTION_COMPOSITE,
          eval_img_option=C.IMG_OPTION_COMPOSITE,
          first_last=False,
          lrcn=False,
          hidden_dim=128,
          num_lstm_layers=1,
          deterministic=True,
          eval_by_pixel=True,
          load_polygon_loss=True,
          load_aux=False,
          late_fusion_regions='none',
          late_fusion_polygon_loss=False,
          late_fusion_embedding_dim=128,
          late_fusion_dropout=0.2,
          load_mode='annual', 
          year_cutoff=None,
          output_pre_logits=False,
          seco_ckpt_path=None,
          aux_subset=False
          ):
    """
    Run the training experiment.

    Args:
        save_dir: Path to save the checkpoints and logs
        exp_name: Name of the experiment
        model: Model name
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
        lr: Learning rate
        pretrained: Whether or not to use the pretrained model
        num_classes: Number of classes
        log_save_interval: Logging saving frequency (in batch)
        distributed_backend: Distributed computing mode
        gradient_clip_val:  Clip value of gradient norm
        train_percent_check: Proportion of training data to use
        max_epochs: Max number of epochs
                patience: number of epochs with no improvement after
                                  which training will be stopped.
        tb_path: Path to global tb folder
        loss_fn: Loss function to use
        weights_summary: Prints a summary of the weights when training begins.
        train_img_option: Options for loading images during training
        eval_img_option: Options for loading images during inference
        first_last: Whether to use the first and last images of each location
        lrcn: Whether to use the Long-term Recurrent Convolutional Network(LSTMs stacked on CNN)
             model should be formatted as "Sequential2DClassifier-modelname"
        load_mode: one of {'scene', 'annual', 'random'} each image loaded represents a scene/year depending on granularity.\
            if 'random', load each time element as scene or annual composite with equal probability. This is IGNORED if composite is True. 
    Returns: None
        late_fusion_regions is one in ['none', 'latlon', 'onehot']

    """
    if exp_name is None:
        exp_name = (f'demo_{datetime.now().strftime("%d%m%Y-%H%M%S")}')
        print(f'No exp name specified. Using {exp_name}')

    # NOTE: Getting this error without this line for late fusion:
    # https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935 
    if load_aux:
        torch.multiprocessing.set_sharing_strategy('file_system')

    args = Args(locals())
    # fixed because split only used at train time
    args['test_split'] = C.VAL_SPLIT

    args['late_fusion'] = late_fusion_regions != 'none' or late_fusion_polygon_loss or load_aux
    args['late_fusion_latlon'] = args['late_fusion'] and late_fusion_regions == 'latlon'
    args['late_fusion_region_embedding'] = args['late_fusion'] and late_fusion_regions == 'onehot'
    args['load_region_embedding'] = args['late_fusion_region_embedding']
    args['late_fusion_aux_feats'] = args['late_fusion'] and args['load_aux']

    args['composite'] = train_img_option in [C.IMG_OPTION_COMPOSITE, C.IMG_OPTION_RANDOM, C.IMG_OPTION_CLOSEST_YEAR, C.IMG_OPTION_FURTHEST_YEAR] and not lrcn

    init_exp_folder(args)
    m = Model(args)

    if regions is not None:
        logging.info('Loading region data')
    else:
        logging.info('Training global model on data from all regions')

    if load_mode not in ['scene', 'annual', 'random', 'annualorscene', 'all']:
        raise Exception('invalid loading mode specified!')
        
    trainer = Trainer(distributed_backend=distributed_backend,
                      gpus=gpus,
                      logger=get_logger(save_dir, exp_name),
                      checkpoint_callback=get_ckpt_callback(save_dir,
                                                            exp_name),
                      early_stop_callback=get_early_stop_callback(patience),
                      weights_save_path=os.path.join(save_dir, exp_name),
                      log_save_interval=log_save_interval,
                      gradient_clip_val=gradient_clip_val,
                      train_percent_check=train_percent_check,
                      weights_summary=weights_summary,
                      max_epochs=max_epochs)
    trainer.fit(m)


def test(ckpt_path,
         gpus=1,
         eval_by_pixel=True,
         load_polygon_loss=True,
         test_split='val',
         output_logits=False,
         output_pre_logits=False,
         **kwargs):
    
    torch.multiprocessing.set_sharing_strategy('file_system')

    m = Model.load_from_checkpoint(ckpt_path,
                                   is_training=False,                     
                                   eval_by_pixel=eval_by_pixel,
                                   load_polygon_loss=load_polygon_loss,
                                   test_split=test_split,
                                   output_logits=output_logits,
                                   output_pre_logits=output_pre_logits,
                                  **kwargs)

    m.hparams['default_save_path'] = os.path.join(
        m.hparams['save_dir'],
        m.hparams['exp_name'],
        f"{m.hparams['test_split']}_results",
        Path(ckpt_path).stem)    
    
    Trainer(gpus=gpus).test(m)

if __name__ == "__main__":
    fire.Fire()
