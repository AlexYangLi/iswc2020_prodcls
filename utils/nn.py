# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: nn.py

@time: 2020/4/26 15:46

@desc:

"""

from tensorflow.keras import optimizers
from transformers import AdamWeightDecay


import tensorflow as tf


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(
        self, initial_learning_rate, decay_schedule_fn, warmup_steps, power=1.0, name=None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


def create_optimizer(init_lr, num_train_steps, num_warmup_steps, end_lr=0.0, optimizer_type="adamw"):
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr, decay_steps=num_train_steps, end_learning_rate=end_lr,
    )
    if num_warmup_steps:
        lr_schedule = WarmUp(
            initial_learning_rate=init_lr, decay_schedule_fn=lr_schedule, warmup_steps=num_warmup_steps,
        )

    optimizer = AdamWeightDecay(
        learning_rate=lr_schedule,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["layer_norm", "bias"],
    )

    return optimizer


def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return optimizers.Adam(learning_rate, clipnorm=5)
    elif op_type == 'adamw':
        return AdamWeightDecay(learning_rate=learning_rate,
                               weight_decay_rate=0.01,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-6,
                               exclude_from_weight_decay=["layer_norm", "bias"])
    elif op_type == 'adamw_2':
        return create_optimizer(init_lr=learning_rate, num_train_steps=9000, num_warmup_steps=0)
    elif op_type == 'adamw_3':
        return create_optimizer(init_lr=learning_rate, num_train_steps=9000, num_warmup_steps=100)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))
