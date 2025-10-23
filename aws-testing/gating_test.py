from utils_neuron import TEST_SUITE, get_gating_configs, create_test_config
from test_cases import GatingTestCase
from axlearn.common.mixture_of_experts import TopKGating, TopKGatingGather, TopKGatingGatherBlockwise, TopKGatingGatherBlockwiseV2
from absl.testing import absltest, parameterized
import os
import jax.numpy as jnp
import unittest
from functools import partial
import jax


@unittest.skipIf(TEST_SUITE != "presubmit", "Skipping tests for suites not equal to presubmit")
class TestGatingOnCpu(GatingTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        jax.config.update('jax_platform_name', 'cpu')
    
    @parameterized.named_parameters(get_gating_configs(test_suite=TEST_SUITE, layer='gating', test=TopKGatingGather, golden=TopKGating, test_device="cpu", golden_device="cpu"))
    def test_fwd_gather(self, cfg):
        self.helper_fwd(cfg)

    @parameterized.named_parameters(get_gating_configs(test_suite=TEST_SUITE, layer='gating', test=TopKGatingGatherBlockwise, golden=None, test_device="cpu"))
    def test_fwd_blockwisegather(self, cfg):
        self.helper_blockwise_gating(cfg)

    @parameterized.named_parameters(get_gating_configs(test_suite=TEST_SUITE, layer='gating', test=TopKGatingGatherBlockwiseV2, golden=TopKGatingGatherBlockwise, test_device="cpu", golden_device="cpu"))
    def test_fwd_blockwisev2(self, cfg):
        self.helper_blockwise_gating_v2_vs_v1(cfg)

class TestDev150bGatingUnit(GatingTestCase):
    def create_cfg(self, test, golden, test_device, golden_device="cpu", layer="gating"):
        return create_test_config(
            layer=layer,
            test=test,
            golden=golden,
            golden_device=golden_device,
            test_device=test_device,
            input_dim=8192,
            hidden_dim=16384,
            n_experts=8,
            n_groups=1,
            top_k=2,
            capacity_factor=2,
            mesh_spec={"fsdp": -1, "model": 16},
            batch=4,
            seq=8192,
            dtype=jnp.bfloat16,
        )[1]
    
    def test_unit_fwd_blockwise(self):
        self.helper_blockwise_gating(self.create_cfg(test=TopKGatingGatherBlockwise, golden=None, test_device="cpu", layer="gating"))

    def test_unit_fwd_blockwisev2(self):
        self.helper_blockwise_gating(self.create_cfg(test=TopKGatingGatherBlockwiseV2, golden=None, test_device="cpu", layer="gating"))

    def test_unit_fwd_blockwisev2_ep(self):
        cfg = create_test_config(
            layer="gating",
            test=TopKGatingGatherBlockwiseV2,
            golden=TopKGatingGatherBlockwise,
            golden_device="cpu",
            test_device="cpu",
            input_dim=1024,
            hidden_dim=4096,
            n_experts=64,
            n_groups=1,
            top_k=1,
            capacity_factor=1,
            mesh_spec={"fsdp": -1, "model": 4, "seq": 4, "expert": 4},
            batch=2,
            seq=32,
            block_size=1,
            dtype=jnp.bfloat16,
        )[1]
        self.helper_blockwise_gating_v2_vs_v1(cfg)

    @unittest.skip("skip gather")
    def test_unit_fwd_gather(self):
        self.helper_fwd(self.create_cfg(test=TopKGatingGather, golden=TopKGating, test_device="cpu", golden_device="cpu", layer="gating"))

class TestDev150bGatingInteg(GatingTestCase):
    def create_cfg(self, test, golden, test_device, golden_device="cpu", layer="gating"):
        return create_test_config(
            layer=layer,
            test=test,
            golden=golden,
            golden_device=golden_device,
            test_device=test_device,
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
    
    def test_integ_fwd_blockwise(self):
        self.helper_blockwise_gating(self.create_cfg(test=TopKGatingGatherBlockwise, golden=None, test_device="neuron", layer="gating"))
    
    def test_integ_fwd_blockwisev2(self):
        self.helper_blockwise_gating(self.create_cfg(test=TopKGatingGatherBlockwiseV2, golden=None, test_device="neuron", layer="gating"))

    def test_integ_fwd_blockwisev2_ep(self):
        cfg = create_test_config(
            layer="gating",
            test=TopKGatingGatherBlockwiseV2,
            golden=TopKGatingGatherBlockwise,
            golden_device="cpu",
            test_device="neuron",
            input_dim=1024,
            hidden_dim=4096,
            n_experts=64,
            n_groups=1,
            top_k=1,
            capacity_factor=1,
            mesh_spec={"fsdp": -1, "model": 4, "seq": 4, "expert": 4},
            batch=2,
            seq=32,
            block_size=1,
            dtype=jnp.bfloat16,
        )[1]
        self.helper_blockwise_gating_v2_vs_v1(cfg)
    
    @unittest.skip("skip gather")
    def test_integ_fwd_gather(self):
        self.helper_fwd(self.create_cfg(test=TopKGatingGather, golden=TopKGating, test_device="neuron", golden_device="cpu", layer="gating"))
