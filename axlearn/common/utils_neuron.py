
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 The Pax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Utils for tests for mixture_of_experts.py"""
from functools import partial
from itertools import product

import jax
import jax.numpy as jnp

from axlearn.common.mixture_of_experts import (
    Top2Gating,
    TransformerFeedForwardMoE,
    TopKGatingGather,
)

from axlearn.common.layers import (
    Dropout,
    StochasticDepth,
    RMSNorm,
)

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

        with jax.default_device(jax.devices(self.test.device)[0]):
            self.test_layer = test.module.instantiate(parent=None)
            self.test_state = self.test_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        with jax.default_device(jax.devices(self.golden.device)[0]):
            self.golden_layer = golden.module.instantiate(parent=None)
            self.golden_state = jax.device_put(self.test_state, jax.devices(self.golden.device)[0])

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
            ),
            # "norm" : RMSNorm.default_config().set(eps=1e-5, forward_dtype=None),
            "dropout" : Dropout.default_config().set(rate=None),
            "stochastic_depth" : StochasticDepth.default_config().set(rate=None)
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
    
    def build_test_configs_integ(self):

        seq_len = (self.params["batch_size"]*self.params["seq_len"])//(self.params["outer_batch"] * self.params["num_groups"])

        return [
            TestConfig(
                setup=[
                    self.build_moe_topkgather_setup(),
                    self.build_moe_topkgather_setup()
                ],
                test=ModuleConfig(TransformerFeedForwardMoE, "neuron"),
                golden=ModuleConfig(TransformerFeedForwardMoE, "cpu"),
                inputs=dict(inputs=jax.random.uniform(
                    jax.random.PRNGKey(1),
                    shape=(self.params["batch_size"], self.params["seq_len"], self.params["input_dim"])
                )),
                loss_fn=lambda x: x.mean(),
                prefix="_moe"
            ),
            TestConfig(
                setup=[
                    self.build_gating_setup(),
                    self.build_gating_setup()
                ],
                test=ModuleConfig(TopKGatingGather, "neuron"),
                golden=ModuleConfig(TopKGatingGather, "cpu"),
                inputs=dict(logits=jax.random.uniform(
                    jax.random.PRNGKey(1),
                    shape=(self.params["outer_batch"], self.params["num_groups"],
                           seq_len, self.params["num_experts"])
                )),
                loss_fn=lambda x: x.load_balance_loss,
                prefix="_gating"
            ),
        ]

    def build_test_configs_unit(self):

        seq_len = (self.params["batch_size"]*self.params["seq_len"])//(self.params["outer_batch"] * self.params["num_groups"])

        return [
            TestConfig(
                setup=[
                    self.build_moe_topkgather_setup(),
                    self.build_moe_top2_setup()
                ],
                test=ModuleConfig(TransformerFeedForwardMoE),
                golden=ModuleConfig(TransformerFeedForwardMoE),
                inputs=dict(inputs=jax.random.uniform(
                    jax.random.PRNGKey(1),
                    shape=(self.params["batch_size"], self.params["seq_len"], self.params["input_dim"])
                )),
                loss_fn=lambda x: x.mean(),
                prefix="_moe"
            ),
            TestConfig(
                setup=[
                    self.build_gating_setup(),
                    self.build_gating_setup()
                ],
                test=ModuleConfig(TopKGatingGather),
                golden=ModuleConfig(Top2Gating),
                inputs=dict(logits=jax.random.uniform(
                    jax.random.PRNGKey(1),
                    shape=(self.params["outer_batch"], self.params["num_groups"],
                           seq_len, self.params["num_experts"])
                )),
                conv_output=partial(_topkgather_to_topk, expert_cap=self.params["expert_capacity"]),
                loss_fn=lambda x: x.load_balance_loss,
                prefix="_gating"
            ),
        ]
    
    def build_grid_space(self):
        # Grid space for testing
        batchs =            [1, 4]
        seqs =              [16, 128]
        input_dims =        [64]
        hidden_dims =       [128]
        num_experts =       [2, 8]
        num_groups =        [1, 4]
        outer_batches =     [1, 2]
        expert_capacities = [2, 1000]

        grid_space = list(product(batchs, seqs, input_dims, hidden_dims, num_experts, num_groups, outer_batches, expert_capacities))

        # Custom Configs
        # b s i h e g ob ec
        grid_space.extend([(2, 100, 64, 128, 2, 1, 1, 5)])

        return grid_space

    
def get_training_configs(is_unit: bool = False):
    builder = TestConfigBuilder()

    test_configs = []

    for (batch, seq, input_dim,  hidden_dim, n_experts, n_groups, out_batch, capacity) in builder.build_grid_space():
        
        if batch % out_batch != 0:
            continue
        
        config = builder.reset()
        config = config.with_dimensions(batch, seq, input_dim)
        config = config.with_expert_settings(
            hidden_dim,
            out_batch,
            n_groups,
            n_experts,
            capacity,
            train_capacity_factor=None
        )
        if is_unit:
            config = config.build_test_configs_unit()
        else:
            config = config.build_test_configs_integ()

        name = f"MoE_b{batch}_s{seq}_i{input_dim}_h{hidden_dim}_e{n_experts}_g{n_groups}_ob{out_batch}_ec{capacity}"
        test_configs.extend([(name + cfg.prefix, cfg) for cfg in config])

    return test_configs