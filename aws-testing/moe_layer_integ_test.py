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

    # @unittest.skip("test fwd skipped as fwd is part of fwd+bwd test")
    # @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGatherBlockwise, golden=TopKGating, test_device="neuron", golden_device="cpu"))
    # def test_fwd_blockwisegather(self, cfg):
    #     self.helper_fwd(cfg)
    
    # @unittest.skip("test fwd skipped as fwd is part of fwd+bwd test")
    # @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGatherBlockwiseV2, golden=TopKGating, test_device="neuron", golden_device="cpu"))
    # def test_fwd_blockwisegatherv2(self, cfg):
    #     self.helper_fwd(cfg)
    
    # @unittest.skip("skip gather")
    # @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGather, golden=TopKGating, test_device="neuron", golden_device="cpu"))
    # def test_fwdbwd_gather(self, cfg: ExperimentConfig):
    #     self.helper_bwd(cfg)

    @unittest.skip("skip v1 tests as v2 is the focus")
    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGatherBlockwise, golden=TopKGating, test_device="neuron", golden_device="cpu"))
    def test_fwdbwd_blockwisegather(self, cfg: ExperimentConfig):
        self.helper_bwd(cfg)

    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGatherBlockwiseV2, golden=TopKGating, test_device="neuron", golden_device="cpu"))
    def test_fwdbwd_blockwisev2(self, cfg: ExperimentConfig):
        self.helper_bwd(cfg)

class TestDev150bInteg(LayerTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_device = 'neuron'
        self.golden_device = 'cpu'
        self.golden = TopKGating
    
    def create_cfg(self, test, golden=None, layer="moe"):
        golden = self.golden if golden is None else golden
        return create_test_config(
            layer=layer,
            test=test,
            golden=golden,
            test_device=self.test_device,
            golden_device=self.golden_device,
            input_dim=8192,
            hidden_dim=16384,
            n_experts=8,
            n_groups=1,
            top_k=2,
            capacity_factor=2,
            mesh_spec={"fsdp": -1, "model": 16},
            batch=8,
            seq=8192,
            dtype=jnp.bfloat16,
        )[1]

    @unittest.skip("skip fwd only as fwd bwd is running")
    def test_fwd_blockwise(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_fwd(self.create_cfg(test=TopKGatingGatherBlockwise))

    @unittest.skip("skip till change merged to compiler")
    def test_fwd_blockwisev2(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_fwd(self.create_cfg(test=TopKGatingGatherBlockwiseV2))
    
    @unittest.skip("skip gather")
    def test_fwd_gather(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_fwd(self.create_cfg(test=TopKGatingGather))
    
    def test_fwdbwd_blockwise(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_bwd(self.create_cfg(test=TopKGatingGatherBlockwise))

    def test_fwdbwd_blockwisev2(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_bwd(self.create_cfg(test=TopKGatingGatherBlockwiseV2))
    
    @unittest.skip("skip gather")
    def test_fwdbwd_gather(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_bwd(self.create_cfg(test=TopKGatingGather))    

if __name__ == "__main__":
    absltest.main()
