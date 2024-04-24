from datetime import datetime
from matplotlib import pyplot as plt

import torch
import torchio as tio
import pytorch_lightning as pl
import pandas as pd
import seaborn as sb
import monai

from model.unet import UNet
from model.lightningmodel import LightningModel
from utils import parse_args

from TrainingPipeline import TrainingPipeline
from attack.attack import GradientInversionAttack
from datamodules import MedicalDecathlonDataModule
sb.set()
plt.rcParams["figure.figsize"] = 12, 8
monai.utils.set_determinism()


# OUR TASK: segment the hippocampus on magnetic resonance images (MRI) of human brains.
# Target: Target: Hippocampus head and body
# Modality: Mono - modal MRI
# Size: 394 3D volumes(263 Training + 131 Testing)
# Source: Vanderbilt University Medical Center

# Credit to GradAttack, Monai, and Medical Decathalon


if __name__ == "__main__":
    args, hparams, attack_hparams = parse_args()


    ################             Data               ###########################
    data = MedicalDecathlonDataModule(
        task="Task04_Hippocampus",
        batch_size=16,
        train_val_ratio=0.8)
    data.prepare_data()
    data.setup()
    print("Training:  ", len(data.train_set))
    print("Validation: ", len(data.val_set))
    print("Test:      ", len(data.test_set))


    #################           Model               ############################
    unet = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=3,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2))
    model = LightningModel(
        net=unet,
        criterion=monai.losses.DiceCELoss(softmax=True),
        learning_rate=1e-2,
        optimizer_class=torch.optim.AdamW,)
    device = model.device


    #################           Pipeline            #############################
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss")
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        max_epochs = 1,
        callbacks=[early_stopping])
    trainer.logger._default_hp_metric = False
    start = datetime.now()
    pipeline = TrainingPipeline(model, data, trainer)
    # Fit
    pipeline.run()


    ################        PLOT VALIDATION RESULTS             #################
    # all_dices = []
    # get_dice = monai.metrics.DiceMetric(include_background=False, reduction="none")
    # with torch.no_grad():
    #     for batch in data.val_dataloader():
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
    #     for batch in data.test_dataloader():
    #          inputs = batch["image"][tio.DATA].to(device)
    #          labels = model.net(inputs).argmax(dim=1, keepdim=True).cpu()
    #          print(len(inputs))
    #          print(len(labels))
    #          break
    # batch_subjects = tio.utils.get_subjects_from_batch(batch)
    # tio.utils.add_images_from_batch(batch_subjects, labels, tio.LabelMap)
    # for subject in batch_subjects:
    #     subject.plot()


    ################            Gradient Inversion             ###################
    trainloader = data.train_dataloader()
    for (idx, batch) in enumerate(trainloader):
        print("batch number" + str(idx))
        inputs = batch["image"][tio.DATA].to(device)
        labels = batch["label"][tio.DATA].to(device)
        #labels = model.net(inputs).argmax(dim=1, keepdim=True).cpu()
        gradients = torch.autograd.grad(
            model.compute_training_step(batch, idx)['loss'],
            model.net.parameters(),
        )

        attack = GradientInversionAttack(model=model,
                                         dm=0, ds=1,
                                         device=device,
                                         loss_metric=monai.losses.DiceCELoss(softmax=True))

        attack.run_attack_batch(batch_inputs=inputs, batch_targets=labels)


        # Attack using reconstructio_algorithms.GradientReconstructor
        # batch_gradients, step_results = model.get_batch_gradients(batch, idx)
        # batch_inputs, batch_targets = step_results[
        #     "transformed_batch"]
        # attack = GradientInversionAttack(model=model,
        #                                  dm=0, ds=1,
        #                                  device=device,
        #                                  loss_metric=monai.losses.DiceCELoss(softmax=True))
        #
        # attack.run_attack_batch(batch_inputs=batch_inputs, batch_targets=batch_targets)


        # Attack using gradientinversion.py

        # print(batch_inputs_transform)
        # print(batch_gradients)
        # print(batch_targets_transform)

        # attack = GradientReconstructor(
        #     model,
        #     ground_truth_inputs=batch_inputs_transform,
        #     ground_truth_gradients=batch_gradients,
        #     ground_truth_labels=batch_targets_transform,
        #     # reconstruct_labels=attack_hparams["reconstruct_labels"],
        #     # num_iterations=10000,
        #     # signed_gradients=True,
        #     # signed_image=attack_hparams["signed_image"],
        #     # boxed=True,
        #     # total_variation=attack_hparams["total_variation"],
        #     # bn_reg=attack_hparams["bn_reg"],
        #     # lr_scheduler=True,
        #     # lr=attack_hparams["attack_lr"],
        #     # attacker_eval_mode=attack_hparams["attacker_eval_mode"],
        #     # BN_exact=attack_hparams["BN_exact"],
        # )
        # attack_trainer = pl.Trainer(
        #     gpus=1,
        #     max_epochs=1)
        # attack_trainer.fit(attack)
        # result = attack.best_guess.detach().to("cpu")








