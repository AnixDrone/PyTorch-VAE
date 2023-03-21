import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import StructuralSimilarityIndexMeasure


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.mask_transform = transforms.RandomErasing(
            p=1.0,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value=0
        )
        #self.inception_score = InceptionScore()
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.ssim = StructuralSimilarityIndexMeasure()
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device
        masked_img = self.mask_transform(real_img)

        results = self.forward(masked_img, labels=labels)
        results[1] = real_img
        train_loss = self.model.loss_function(*results,
                                              # al_img.shape[0]/ self.num_train_imgs,
                                              M_N=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()},
                      sync_dist=True, on_epoch=True, on_step=False)
        self.log('lpips',
                 self.lpips(results[0], real_img),
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 )
        self.log('ssim',
                 self.ssim(results[0], real_img),
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 )

        return train_loss['loss']

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        real_img, labels = batch
        self.curr_device = real_img.device
        # masked_img = self.mask_transform(real_img)
        print('NESTO')
        results = self.forward(real_img, labels=labels)
        return results

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device
        masked_img = self.mask_transform(real_img)

        results = self.forward(masked_img, labels=labels)
        
        results[1] = real_img
        val_loss = self.model.loss_function(*results,
                                            # real_img.shape[0]/ self.num_val_imgs,
                                            M_N=1.0,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)
        
        
        vutils.save_image(results[0],
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        tensor_board_imgs = vutils.make_grid(results[0],
                                             normalize=True, 
                                             nrow=12)

        self.logger.experiment.add_image("Reconstructions", 
                                         tensor_board_imgs, 
                                         global_step=self.current_epoch)
        
        #self.inception_score.update(results[0])
        self.log('val_lpips',
                 self.lpips(results[0], real_img),
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 )
        self.log('val_ssim',
                 self.ssim(results[0], real_img),
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 )
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()},
                      sync_dist=True)
        
    def test_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device
        masked_img = self.mask_transform(real_img)

        results = self.forward(masked_img, labels=labels)
        
        results[1] = real_img
        val_loss = self.model.loss_function(*results,
                                            # real_img.shape[0]/ self.num_val_imgs,
                                            M_N=1.0,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)
        
        
        vutils.save_image(results[0],
                          os.path.join(self.logger.log_dir,
                                       "test_Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        tensor_board_imgs = vutils.make_grid(results[0],
                                             normalize=True, 
                                             nrow=12)

        self.logger.experiment.add_image("test_Reconstructions", 
                                         tensor_board_imgs, 
                                         global_step=self.current_epoch)
        
        #self.inception_score.update(results[0])
        self.log('test_lpips',
                 self.lpips(results[0], real_img),
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 )
        self.log('test_ssim',
                 self.ssim(results[0], real_img),
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 )
        self.log_dict({f"test_{key}": val.item() for key, val in val_loss.items()},
                      sync_dist=True)
        #return val_loss['loss']

    def on_validation_end(self) -> None:
        #self.log('val_inception_score',
        #         self.inception_score.compute(),
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=True,
        #         logger=True)
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(
            iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        #recons = self.model.generate(test_input, labels=test_label)
        # vutils.save_image(recons.data,
        #                   os.path.join(self.logger.log_dir,
        #                                "Reconstructions",
        #                                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                   normalize=True,
        #                   nrow=12)
        # tensor_board_imgs = vutils.make_grid(
        #     recons.data, normalize=True, nrow=12)

        # self.logger.experiment.add_image(
        #     f"Reconstructions", tensor_board_imgs, global_step=self.current_epoch)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
