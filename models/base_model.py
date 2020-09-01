# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: base_model.py

@time: 2020/5/21 7:27

@desc:

"""

import os
import math

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from callbacks import *


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.callbacks = []
        self.model = None
        self.swa_model = None
        self.polyak_model, self.temp_polyak_model = None, None
        self.earlystop_callback = None

    def add_model_checkpoint(self):
        self.callbacks.append(ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, '{}.hdf5'.format(self.config.exp_name)),
            monitor=self.config.checkpoint_monitor,
            save_best_only=self.config.checkpoint_save_best_only,
            save_weights_only=self.config.checkpoint_save_weights_only,
            mode=self.config.checkpoint_save_weights_mode,
            verbose=self.config.checkpoint_verbose
        ))
        print('Logging Info - Callback Added: ModelCheckPoint...')

    def add_early_stopping(self):
        self.earlystop_callback = EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            mode=self.config.early_stopping_mode,
            patience=self.config.early_stopping_patience,
            verbose=self.config.early_stopping_verbose
        )
        self.callbacks.append(self.earlystop_callback)
        print('Logging Info - Callback Added: EarlyStopping...')

    def add_clr(self, kind, min_lr, max_lr, cycle_length):
        """
        add cyclic learning rate schedule callback
        :param kind: add what kind of clr, 0: the original cyclic lr, 1: the one introduced in FGE, 2: the one
                     introduced in swa
        """
        if kind == 0:
            self.callbacks.append(CyclicLR(base_lr=min_lr, max_lr=max_lr, step_size=cycle_length/2, mode='triangular2',
                                           plot=True, save_plot_prefix=self.config.exp_name))
        elif kind == 1:
            self.callbacks.append(CyclicLR_1(min_lr=min_lr, max_lr=max_lr, cycle_length=cycle_length, plot=True,
                                             save_plot_prefix=self.config.exp_name))
        elif kind == 2:
            self.callbacks.append(CyclicLR_2(min_lr=min_lr, max_lr=max_lr, cycle_length=cycle_length, plot=True,
                                             save_plot_prefix=self.config.exp_name))
        else:
            raise ValueError('param `kind` not understood : {}'.format(kind))
        print('Logging Info - Callback Added: CLR_{}...'.format(kind))

    def add_sgdr(self, min_lr, max_lr, cycle_length):
        self.callbacks.append(SGDR(min_lr=min_lr, max_lr=max_lr, cycle_length=cycle_length,
                                   save_plot_prefix=self.config.exp_name))
        print('Logging Info - Callback Added: SGDR...')

    def add_swa(self, with_clr, min_lr=None, max_lr=None, cycle_length=None, swa_start=5):
        if with_clr:
            self.callbacks.append(SWAWithCLR(self.swa_model, self.config.checkpoint_dir, self.config.exp_name,
                                             min_lr=min_lr, max_lr=max_lr, cycle_length=cycle_length,
                                             swa_start=swa_start))
        else:
            self.callbacks.append(SWA(self.swa_model, self.config.checkpoint_dir, self.config.exp_name,
                                      swa_start=swa_start))
        print('Logging Info - Callback Added: SWA with {}...'.format('clr' if with_clr else 'constant lr'))

    def add_polyak(self, avg_type, polyak_start=5):
        self.callbacks.append(PolyakAverage(self.polyak_model, self.temp_polyak_model, self.config.checkpoint_dir,
                                            self.config.exp_name, avg_type, self.config.early_stopping_patience,
                                            polyak_start))
        print('Logging Info - Callback Added: Polyak Average Ensemble...')

    def add_hor(self, hor_start=5):
        self.callbacks.append(HorizontalEnsemble(self.config.checkpoint_dir, self.config.exp_name, hor_start))
        print('Logging Info - Callback Added: Horizontal Ensemble...')

    def add_sse(self, max_lr, cycle_length, sse_start):
        self.callbacks.append(SnapshotEnsemble(self.config.checkpoint_dir, self.config.exp_name,
                                               max_lr=max_lr, cycle_length=cycle_length, snapshot_start=sse_start))
        print('Logging Info - Callback Added: Snapshot Ensemble...')

    def add_fge(self, min_lr, max_lr, cycle_length, fge_start):
        self.callbacks.append(FGE(self.config.checkpoint_dir, self.config.exp_name, min_lr=min_lr, max_lr=max_lr,
                                  cycle_length=cycle_length, fge_start=fge_start))
        print('Logging Info - Callback Added: Fast Geometric Ensemble...')

    def add_warmup(self, lr=5e-5, min_lr=1e-5):
        self.callbacks.append(WarmUp(lr, min_lr))
        print('Logging Info - Callback Added: WarmUp....')

    def init_callbacks(self, data_size):
        cycle_length = 6 * math.floor(data_size / self.config.batch_size)

        if 'modelcheckpoint' in self.config.callbacks_to_add:
            self.add_model_checkpoint()
        if 'earlystopping' in self.config.callbacks_to_add:
            self.add_early_stopping()
        if 'clr' in self.config.callbacks_to_add:
            self.add_clr(kind=0, min_lr=self.config.min_lr, max_lr=self.config.max_lr, cycle_length=cycle_length)
        if 'sgdr' in self.config.callbacks_to_add:
            self.add_sgdr(min_lr=self.config.min_lr, max_lr=self.config.max_lr, cycle_length=cycle_length)
        if 'clr_1' in self.config.callbacks_to_add:
            self.add_clr(kind=1, min_lr=self.config.min_lr, max_lr=self.config.max_lr, cycle_length=cycle_length)
        if 'clr_2' in self.config.callbacks_to_add:
            self.add_clr(kind=2, min_lr=self.config.min_lr, max_lr=self.config.max_lr, cycle_length=cycle_length)
        if 'swa' in self.config.callbacks_to_add:
            self.add_swa(with_clr=False, swa_start=self.config.swa_start)
        if 'swa_clr' in self.config.callbacks_to_add:
            self.add_swa(with_clr=True, min_lr=self.config.min_lr, max_lr=self.config.max_lr, cycle_length=cycle_length,
                         swa_start=self.config.swq_clr_start)
        if 'sse' in self.config.callbacks_to_add:
            self.add_sse(max_lr=self.config.max_lr, cycle_length=cycle_length, sse_start=self.config.sse_start)
        if 'fge' in self.config.callbacks_to_add:
            self.add_fge(min_lr=self.config.min_lr, max_lr=self.config.max_lr, cycle_length=cycle_length,
                         fge_start=self.config.fge_start)
        if 'hor' in self.config.callbacks_to_add:
            self.add_hor(hor_start=self.config.swa_start)
        if 'polyak_avg' in self.config.callbacks_to_add:
            self.add_polyak('avg', self.config.polyak_start)
        if 'polyak_linear' in self.config.callbacks_to_add:
            self.add_polyak('linear', self.config.polyak_start)
        if 'polay_exp' in self.config.callbacks_to_add:
            self.add_polyak('exp', self.config.polyak_start)
        if 'warmup' in self.config.callbacks_to_add:
            self.add_warmup(lr=self.config.max_lr, min_lr=self.config.min_lr)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def load_model(self, filename):
        # we only save model's weight instead of the whole model
        self.model.load_weights(filename)

    def load_best_model(self):
        print('Logging Info - Loading model checkpoint: %s.hdf5\n' % self.config.exp_name)
        self.load_model(os.path.join(self.config.checkpoint_dir, '{}.hdf5'.format(self.config.exp_name)))
        print('Logging Info - Model loaded')

    def load_swa_model(self, swa_type='swa'):
        print('Logging Info - Loading SWA model checkpoint: %s_%s.hdf5\n' % (self.config.exp_name, swa_type))
        self.load_model(os.path.join(self.config.checkpoint_dir, '%s_%s.hdf5' % (self.config.exp_name, swa_type)))
        print('Logging Info - SWA Model loaded')

    def load_polyak_model(self, polyak_type='swa'):
        print('Logging Info - Loading Polyak Average model checkpoint: %s_polyak_%s.hdf5\n' % (self.config.exp_name, polyak_type))
        self.load_model(os.path.join(self.config.checkpoint_dir, '%s_polyak_%s.hdf5' % (self.config.exp_name, polyak_type)))
        print('Logging Info - Polyak Model loaded')

    def add_metric_callback(self, valid_generator):
        raise NotImplementedError

    def train(self, train_generator, valid_generator):
        self.callbacks = []
        self.add_metric_callback(valid_generator=valid_generator)
        self.init_callbacks(train_generator.data_size)

        print('Logging Info - Start training...')
        self.model.fit(x=train_generator, epochs=self.config.n_epoch, callbacks=self.callbacks)
        print('Logging Info - Training end...')

    def return_trained_epoch(self):
        # https://stackoverflow.com/questions/49852241/return-number-of-epochs-for-earlystopping-callback-in-keras
        if self.earlystop_callback:
            stopped_epoch = self.earlystop_callback.stopped_epoch + 1
            if stopped_epoch == 1:
                return self.config.n_epoch
            else:
                return stopped_epoch
        else:
            return self.config.n_epoch

    def summary(self):
        self.model.summary()
