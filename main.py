from matplotlib import pyplot as plt

import torch
import torchio as tio
import pytorch_lightning as pl
import pandas as pd
import seaborn as sb
import monai
from torchio import ScalarImage

from attacks.metrics import psnr, ncc, ssim  # , ssim
from model.unet import UNet
from model.lightningmodel import LightningModel

from TrainingPipeline import TrainingPipeline
from attacks.attack import GradientInversionAttack
from datamodules import MedicalDecathlonDataModule


sb.set()
plt.rcParams["figure.figsize"] = 12, 8
monai.utils.set_determinism()

plots = []
def plotImage(rec_batch_subjects, opt):
    ground_truth = None
    for subject, data in zip(rec_batch_subjects, opt):
        one_image = subject.get_first_image()
        kwargs = {'tensor': data, 'affine': one_image.affine}
        if 'filename' in one_image:
            kwargs['filename'] = one_image['filename']
        image = ScalarImage(**kwargs)
        transform = tio.Compose([datamodule.get_preprocessing_transform(), datamodule.get_augmentation_transform()])
        subject.add_image(transform(image), 'reconstruction')

    for subject in rec_batch_subjects:
        ground_truth = subject.get_images()[0].data
        plots.append(subject)
    return ground_truth

def logMetrics(ground_truth, opt):
    print("PSNR: " + str(psnr(opt, ground_truth)))
    print("NCC: " + str(ncc(opt, ground_truth)))
    print("SSIM: " + str(ssim(opt, ground_truth)))


if __name__ == "__main__":
    ################             Data               ###########################
    datamodule = MedicalDecathlonDataModule(
        task="Task04_Hippocampus",
        batch_size=1,
        train_val_ratio=0.8)
    datamodule.prepare_data()
    datamodule.setup()
    print("Training:  ", len(datamodule.train_set))
    print("Validation: ", len(datamodule.val_set))
    print("Test:      ", len(datamodule.test_set))


    #################           Model               ############################
    unet = UNet(
        dimensions=3,
        in_channels=1,
        channels=(8, 16, 32, 64),
        out_channels=3,
        strides=(2, 2, 2))
    model = LightningModel(
        net=unet,
        criterion=monai.losses.DiceCELoss(softmax=True),
        learning_rate=1e-2,
        optimizer_class=torch.optim.AdamW,)
    device = model.device


    #################           Pipeline and Logs          #############################
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss")

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1,
        precision=16,
        callbacks=[early_stopping],)


    pipeline = TrainingPipeline(model, datamodule, trainer)
    # Fit
    pipeline.run()


    ################        PLOT VALIDATION RESULTS             #################
    # all_dices = []
    # get_dice = monai.metrics.DiceMetric(include_background=False, reduction="none")
    # with torch.no_grad():
    #     for batch in datamodule.val_dataloader():
    #         inputs, targets = model.prepare_batch(batch)
    #         logits = model.net(inputs.to(device))
    #         labels = logits.argmax(dim=1)
    #         labels_one_hot = torch.nn.functional.one_hot(labels).permute(0, 4, 1, 2, 3)
    #         get_dice(labels_one_hot.to(device), targets.to(device))
    #     metric = get_dice.aggregate()
    #     get_dice.reset()
    #     all_dices.append(metric)
    # all_dices = torch.cat(all_dices)
    # records = []
    # for ant, post in all_dices:
    #     records.append({"Dice": ant, "Label": "Anterior"})
    #     records.append({"Dice": post, "Label": "Posterior"})
    # df = pd.DataFrame.from_records(records)
    # ax = sb.stripplot(x="Label", y="Dice", data=df, size=10, alpha=0.5)


    ###################                TEST                 #####################
    # with torch.no_grad():
    #     print("TEST-------------------------")
    #     for batch in datamodule.test_dataloader():
    #          inputs = batch["image"][tio.DATA].to(device)
    #          labels = model.net(inputs).argmax(dim=1, keepdim=True).cpu()
    #          break
    # batch_subjects = tio.utils.get_subjects_from_batch(batch)
    # tio.utils.add_images_from_batch(batch_subjects, labels, tio.LabelMap)
    # for subject in batch_subjects:
    #     subject.plot()


    ################            Gradient Inversion             ###################
    mean = 0
    std = 1

    trainloader = datamodule.train_dataloader()
    for (idx, batch) in enumerate(trainloader):
        if(idx<2):
            continue
        if(idx==3):
            break
        attack = GradientInversionAttack(pipeline,
                                         dm=mean, ds=std,
                                         device=device,
                                         loss_metric=model.criterion)
        print("-------- Running vanilla attack")
        opt, stats = attack.run_attack_batch(batch, vanilla = True)
        logMetrics(plotImage(tio.utils.get_subjects_from_batch(batch), opt), opt)

        print("-------- Running attack with DP")
        opt, stats = attack.run_attack_batch(batch, grad_clip=True, grad_prune=False, grad_noise=True, keep_pruned=False)
        logMetrics(plotImage(tio.utils.get_subjects_from_batch(batch), opt), opt)

        print("-------- Running attack with gradient pruning")
        opt, stats = attack.run_attack_batch(batch, grad_clip=False, grad_prune=True, grad_noise=False, keep_pruned=False)
        logMetrics(plotImage(tio.utils.get_subjects_from_batch(batch), opt), opt)

        print("-------- Running attack with pruning and DP")
        opt, stats = attack.run_attack_batch(batch, grad_clip=True, grad_prune=True, grad_noise=True, keep_pruned=False)
        logMetrics(plotImage(tio.utils.get_subjects_from_batch(batch), opt), opt)

        print("-------- Running attack with pruning and DP")
        opt, stats = attack.run_attack_batch(batch, grad_clip=True, grad_prune=True, grad_noise=True, keep_pruned=True)
        logMetrics(plotImage(tio.utils.get_subjects_from_batch(batch), opt), opt)

        for subject in plots:
            subject.plot()









