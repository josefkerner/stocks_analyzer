import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

class TimeSeriesTransformer:
    def __init__(self):
        pass

    def transform_dataset(self, data: pd.DataFrame):
        '''

        :param data:
        :return:
        '''
        # add time index as dates converted to continuous ids
        data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
        data["month"] = data["date"].dt.month.astype(str).astype("category")
        data["year"] = data["date"].dt.year.astype(str).astype("category")


    def get_dataset(self, data: pd.DataFrame):
        '''

        :param data:
        :return:
        '''
        max_prediction_length = 6
        max_encoder_length = 24
        training_cutoff = data["time_idx"].max() - max_prediction_length

        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="volume",
            group_ids=["agency", "sku"],
            min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["ti"],
            static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
            time_varying_known_categoricals=["special_days", "month"],
            variable_groups={"special_days": special_days},
            # group of categorical variables can be treated as one variable
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "volume",
                "close"
                "log_volume",
                "industry_volume",
                "soda_volume",
                "avg_max_temp",
                "avg_volume_by_agency",
                "avg_volume_by_sku",
            ],
            target_normalizer=GroupNormalizer(
                groups=["agency", "sku"], transformation="softplus"
            ),  # use softplus and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        validation = TimeSeriesDataSet.from_dataset(training, data, predict=True,
                                                    stop_randomization=True
                                                    )

        # create dataloaders for model
        batch_size = 128  # set this between 32 to 128
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
        return training, validation, train_dataloader, val_dataloader

    def train(self, data: pd.DataFrame):
        '''
        Trains a google time series transformer model
        :return:
        '''
        # load data
        training, validation, train_dataloader, val_dataloader = self.get_dataset(data)

        # configure network and trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4,
                                            patience=10,
                                            verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        trainer = pl.Trainer(
            max_epochs=50,
            accelerator="cpu",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            limit_train_batches=50,  # coment in for training, running validation every 30 batches
            callbacks=[lr_logger, early_stop_callback],
        )

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(),
            log_interval=10,
            # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            optimizer="Ranger",
            reduce_on_plateau_patience=4,
        )
        print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

        # fit network
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )



