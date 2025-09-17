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

    # TODO

class TestDevSwitchBaseInteg(LayerTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_device = 'neuron'
        self.golden_device = 'cpu'
        self.golden = TopKGating

    def test_fwdbwd_transformer(self):
        jax.config.update('jax_platform_name', 'neuron')
        cfg = create_test_config(
            layer="transformer",
            test=TopKGatingGatherBlockwiseV2,
            golden=None,
            test_device=self.test_device,
            golden_device=self.golden_device,
            input_dim=1024,
            hidden_dim=4096,
            n_experts=64,
            n_groups=1,
            top_k=2,
            capacity_factor=2,
            mesh_spec={"expert": 4, "model": 4, "fsdp": 1, "seq": 4},
            batch=4,
            seq=2048,
            dtype=jnp.bfloat16,
        )[1]
        self.helper_bwd(cfg)

if __name__ == "__main__":
    absltest.main()