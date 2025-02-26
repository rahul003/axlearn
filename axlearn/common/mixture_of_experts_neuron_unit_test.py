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
"""Unit Test for mixture_of_experts.py"""
from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.mixture_of_experts import (
    Top2Gating,
    TopKGating,
    TransformerFeedForwardMoE,
    TopKGatingGather,
)
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase

jax.config.update('jax_platform_name', 'cpu')

MODULE_UNIT_TEST_ATOL=1e-6
MODULE_UNIT_TEST_RTOL=1e-3

class ModuleConfig():
    def __init__(self, module = None, device = "cpu", mesh = (1,)):
        assert module is not None
        self.module = module.default_config().set(name="test")
        self.device = device
        self.mesh = mesh

class TestConfig():
    def __init__(self, setup, test: ModuleConfig, golden: ModuleConfig = None, inputs: dict = None, loss_fn = None, conv_output = None, prefix = None):
        self.setup = setup
        self.test = test
        self.golden = golden if golden is not None else test
        self.inputs = inputs
        self.loss_fn = loss_fn
        self.conv_output = conv_output
        self.prefix = prefix

        for spec, val in setup[0].items():
            setattr(self.test.module, spec, val)
        
        for spec, val in setup[1].items():
            setattr(self.golden.module, spec, val)
        
        self.test_layer = test.module.instantiate(parent=None)

        self.golden_layer = golden.module.instantiate(parent=None)

        self.test_state = self.test_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        self.golden_state = self.test_state
    
def _topkgather_to_topk(output, expert_cap):
    tok_perm_idx, expert_index, exp_aff_mask = output.combine_tensor

    O, G, S, _ = tok_perm_idx.shape
    E = exp_aff_mask.shape[-1]

    exp_aff = jnp.take_along_axis(exp_aff_mask, expert_index, axis=-1)

    base = jnp.zeros((O, G, S, E * expert_cap), dtype=exp_aff_mask.dtype)

    idx_O, idx_G, idx_S = jnp.meshgrid(
        jnp.arange(O), 
        jnp.arange(G), 
        jnp.arange(S), 
        indexing='ij'
    )

    output_tensor = base.at[idx_O[..., None], idx_G[..., None], idx_S[..., None], tok_perm_idx].add(exp_aff)
    output_tensor = output_tensor.reshape(O, G, S, E, expert_cap)

    dispatch_tensor = output_tensor.astype(bool)

    return TopKGatingGather.Output(
        combine_tensor=output_tensor,
        dispatch_tensor=dispatch_tensor,
        load_balance_loss=output.load_balance_loss,
        router_z_loss=output.router_z_loss
    )

class TestConfigBuilder:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.params = {
            "batch_size": 1,
            "seq_len": 32,
            "input_dim": 4,
            "hidden_dim": 4,
            "num_experts": 4,
            "num_groups": 1,
            "outer_batch": 1,
            "expert_capacity": 1000,
            "train_capacity_factor": None
        }
        return self
    
    def with_dimensions(self, batch_size, seq_len, input_dim):
        self.params.update({
            "batch_size": batch_size,
            "seq_len": seq_len,
            "input_dim": input_dim
        })
        return self
    
    def with_expert_settings(self, hidden_dim, outer_batch, num_groups, num_experts, expert_capacity, train_capacity_factor=None):
        self.params.update({
            "hidden_dim": hidden_dim,
            "outer_batch" : outer_batch,
            "num_groups": num_groups,
            "num_experts": num_experts,
            "expert_capacity": expert_capacity,
            "train_capacity_factor": train_capacity_factor
        })
        return self
    
    def build_moe_topkgather_setup(self):
        return {
            "input_dim": self.params["input_dim"],
            "hidden_dim": self.params["hidden_dim"],
            "num_experts": self.params["num_experts"],
            "num_groups": self.params["num_groups"],
            "outer_batch": self.params["outer_batch"],
            "gating": TopKGatingGather.default_config().set(
                name="gating",
                expert_capacity=self.params["expert_capacity"],
                train_capacity_factor=self.params["train_capacity_factor"]
            )
        }
    
    def build_moe_top2_setup(self):
        return {
            "input_dim": self.params["input_dim"],
            "hidden_dim": self.params["hidden_dim"],
            "num_experts": self.params["num_experts"],
            "num_groups": self.params["num_groups"],
            "outer_batch": self.params["outer_batch"],
            "gating": Top2Gating.default_config().set(
                name="gating",
                expert_capacity=self.params["expert_capacity"],
                train_capacity_factor=self.params["train_capacity_factor"]
            )
        }
    
    def build_gating_setup(self):
        return {
            "expert_capacity": self.params["expert_capacity"],
            "num_experts": self.params["num_experts"],
            "train_capacity_factor": self.params["train_capacity_factor"]
        }
    
    def build_test_configs_moe(self):
        return [
            TestConfig(
                setup=[
                    self.build_moe_topkgather_setup(),
                    self.build_moe_top2_setup()
                ],
                test=ModuleConfig(TransformerFeedForwardMoE, "cpu"),
                golden=ModuleConfig(TransformerFeedForwardMoE, "cpu"),
                inputs=dict(inputs=jax.random.uniform(
                    jax.random.PRNGKey(1),
                    shape=(self.params["batch_size"], self.params["seq_len"], self.params["input_dim"])
                )),
                loss_fn=lambda x: x.mean(),
                prefix="_unit_test"
            ),
        ]

    def build_test_configs_topk(self):

        seq_len = (self.params["batch_size"]*self.params["seq_len"])//(self.params["outer_batch"] * self.params["num_groups"])

        return [
            TestConfig(
                setup=[
                    self.build_gating_setup(),
                    self.build_gating_setup()
                ],
                test=ModuleConfig(TopKGatingGather, "cpu"),
                golden=ModuleConfig(TopKGating, "cpu"),
                inputs=dict(logits=jax.random.uniform(
                    jax.random.PRNGKey(1),
                    shape=(self.params["outer_batch"], self.params["num_groups"],
                           seq_len, self.params["num_experts"])
                )),
                conv_output=partial(_topkgather_to_topk, expert_cap=self.params["expert_capacity"]),
                prefix="_unit_test"
            ),
        ]
    
def _get_training_configs():
    builder = TestConfigBuilder()

    batchs =            [1, 4]
    seqs =              [16, 128]
    input_dims =        [64]
    hidden_dims =       [128]
    num_experts =       [2, 8]
    num_groups =        [1, 4]
    outer_batches =     [1, 2]
    expert_capacities = [2, 1000]

    test_configs = []

    for (batch, seq, input_dim,  hidden_dim, n_experts, n_groups, out_batch, capacity) in product(
         batchs, seqs, input_dims, hidden_dims, num_experts, num_groups, outer_batches, expert_capacities):
        
        if batch % out_batch != 0:
            continue
        
        config = builder.reset().with_dimensions(batch, seq, input_dim).with_expert_settings(
            hidden_dim,
            out_batch,
            n_groups,
            n_experts,
            capacity,
            train_capacity_factor=None
        ).build_test_configs_moe()
        name = f"MoE_b{batch}_s{seq}_i{input_dim}_h{hidden_dim}_e{n_experts}_g{n_groups}_ob{out_batch}_ec{capacity}"
        test_configs.extend([(name + cfg.prefix, cfg) for cfg in config])
        
        config = builder.build_test_configs_topk()
        name = f"Gating_b{batch}_s{seq}_i{input_dim}_h{hidden_dim}_e{n_experts}_g{n_groups}_ob{out_batch}_ec{capacity}"
        test_configs.extend([(name + cfg.prefix, cfg) for cfg in config])

    return test_configs

def _get_training_configs_bwd():
    builder = TestConfigBuilder()

    batchs =            [1, 4]
    seqs =              [16, 128]
    input_dims =        [64]
    hidden_dims =       [128]
    num_experts =       [2, 8]
    num_groups =        [1, 4]
    outer_batches =     [1, 2]
    expert_capacities = [2, 1000]

    test_configs = []

    for (batch, seq, input_dim,  hidden_dim, n_experts, n_groups, out_batch, capacity) in product(
         batchs, seqs, input_dims, hidden_dims, num_experts, num_groups, outer_batches, expert_capacities):
        
        if batch % out_batch != 0:
            continue

        config = builder.reset().with_dimensions(batch, seq, input_dim).with_expert_settings(
                    hidden_dim,
                    out_batch,
                    n_groups,
                    n_experts,
                    capacity,
                    train_capacity_factor=None
                ).build_test_configs_moe()
        name = f"b{batch}_s{seq}_i{input_dim}_h{hidden_dim}_e{n_experts}_g{n_groups}_ob{out_batch}_ec{capacity}"
        test_configs.extend([(name + cfg.prefix, cfg) for cfg in config])

    return test_configs

# pylint: disable=no-self-use,protected-access
class TestImplCorrectness(TestCase):

    def _fwd_call(self, layer, state, inputs):
        return F(
                layer,
                is_training=True,
                prng_key=jax.random.PRNGKey(123),
                state=state,
                inputs=inputs,
        )

    @parameterized.named_parameters(_get_training_configs())
    def test_fwd_correctness(self, cfg: TestConfig):

        @partial(jax.jit, backend=cfg.test.device)
        def test_fwd_call():
            test_output, _ = self._fwd_call(cfg.test_layer, cfg.test_state, cfg.inputs)
            return test_output

        @partial(jax.jit, backend=cfg.golden.device)
        def golden_fwd_call():
            golden_output, _ =  self._fwd_call(cfg.golden_layer, cfg.golden_state, cfg.inputs)
            return golden_output
        
        test_output = test_fwd_call()
        golden_output = golden_fwd_call()

        if cfg.conv_output != None:
            test_output = cfg.conv_output(test_output)
        
        # Transfer results to CPU before comparison
        self.assertNestedAllClose(jax.device_get(test_output), jax.device_get(golden_output))

    @parameterized.named_parameters(_get_training_configs_bwd())
    def test_bwd_correctness(self, cfg: TestConfig):

        @partial(jax.jit, backend=cfg.test.device)
        def test_bwd_call(state):
            def loss_fn(state):
                test_output, _ = self._fwd_call(cfg.test_layer, state, cfg.inputs)
                return cfg.loss_fn(test_output)
            
            loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state)
            return loss, grads

        @partial(jax.jit, backend=cfg.golden.device)
        def golden_bwd_call(state):
            def loss_fn(state):
                golden_output, _ = self._fwd_call(cfg.golden_layer, state, cfg.inputs)
                return cfg.loss_fn(golden_output)
            
            loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state)
            return loss, grads

        test_loss, test_grads = test_bwd_call(cfg.test_state)
        golden_loss, golden_grads = golden_bwd_call(cfg.golden_state)

        # Transfer results to CPU before comparison
        test_loss = jax.tree_map(jax.device_get, test_loss)
        golden_loss = jax.tree_map(jax.device_get, golden_loss)
        test_grads = jax.tree_map(jax.device_get, test_grads)
        golden_grads = jax.tree_map(jax.device_get, golden_grads)
        
        self.assertNestedAllClose(test_loss, golden_loss)
        self.assertNestedAllClose(test_grads, golden_grads)

if __name__ == "__main__":
    absltest.main()
