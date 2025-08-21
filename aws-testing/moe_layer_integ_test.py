# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 The Pax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Integration Test for mixture_of_experts.py"""
from functools import partial
import unittest
import os
import jax
import math
import jax.numpy as jnp
from jax.sharding import Sharding
from jax_neuronx.experimental import debug_callback
from axlearn.common.utils import PartitionSpec
from absl.testing import absltest, parameterized
from axlearn.common.mixture_of_experts import TopKGatingGather, TopKGating, TopKGatingGatherBlockwise, TopKGatingGatherBlockwiseV2
from axlearn.common.test_utils import TestCase
from test_cases import LayerTestCase
from utils_neuron import ExperimentConfig, create_test_config, get_training_configs, get_gating_configs, TEST_SUITE
import os

class TestLayerOnTrn(LayerTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        jax.config.update('jax_platform_name', 'neuron')
    
    def tearDown(self):
        from jax._src import xla_bridge as xb
        xb._clear_backends()
        return super().tearDown()

    def test_single_fast_test(self):
        pass  # Single fast test - always passes

    



if __name__ == "__main__":
    absltest.main()