# Copyright © 2025 Amazon Inc.

"""Neuron ring attention kernel wrapper for context parallelism."""

import os
from functools import partial

import jax
from absl import logging
from jax import custom_vjp

from axlearn.common.attention_bias import BaseAttentionBias, CausalAttentionBias, split
from axlearn.common.flash_attention.common import BaseFlashAttention, repeat_kv_heads
from axlearn.common.utils import Nested, Tensor

try:
    from neuronxcc.nki.kernels.attention import (
        ring_attention_spmd_fwd,
        ring_attention_spmd_bwd,
    )
    _HAS_RING_KERNEL = True
except ImportError:
    _HAS_RING_KERNEL = False


def _get_replica_groups_and_workers(mesh, axis_name: str):
    """Compute replica groups and num_workers for ring attention."""
    axis_index = mesh.axis_names.index(axis_name)
    num_workers = mesh.shape[axis_index]

    total_devices = len(mesh.devices.flat)
    devices_per_group = num_workers
    replica_groups = []

    for i in range(total_devices // devices_per_group):
        group = list(range(i * devices_per_group, (i + 1) * devices_per_group))
        replica_groups.append(group)

    return replica_groups, num_workers


@partial(custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def ring_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    prng_key: Tensor,
    replica_groups: list,
    num_workers: int,
    causal: bool,
    softmax_scale: float,
    dropout_rate: float,
):
    """Ring attention forward pass with custom vjp."""
    out, _ = _ring_forward(query, key, value, prng_key, replica_groups, num_workers, causal, softmax_scale, dropout_rate)
    return out


def _stripe_sequence(x: Tensor, num_workers: int) -> Tensor:
    """Stripe sequence dimension across workers.
    
    Converts [batch, seq, ...] to [batch, seq//num_workers, num_workers, ...]
    then transposes to [batch, num_workers, seq//num_workers, ...].
    """
    batch, seq_len = x.shape[0], x.shape[1]
    seq_per_worker = seq_len // num_workers
    # Reshape and transpose to stripe
    x = x.reshape(batch, seq_per_worker, num_workers, *x.shape[2:])
    x = x.transpose(0, 2, 1, *range(3, x.ndim))
    return x


def _unstripe_sequence(x: Tensor, num_workers: int) -> Tensor:
    """Unstripe sequence dimension from workers.
    
    Converts [batch, num_workers, seq//num_workers, ...] back to [batch, seq, ...].
    """
    batch = x.shape[0]
    # Transpose and reshape to unstripe
    x = x.transpose(0, 2, 1, *range(3, x.ndim))
    x = x.reshape(batch, -1, *x.shape[3:])
    return x


def _ring_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    prng_key: Tensor,
    replica_groups: list,
    num_workers: int,
    causal: bool,
    softmax_scale: float,
    dropout_rate: float,
):
    """Ring attention forward pass."""
    # Stripe inputs across workers only when causal mask is used
    if causal:
        query = _stripe_sequence(query, num_workers)
        key = _stripe_sequence(key, num_workers)
        value = _stripe_sequence(value, num_workers)
    
    q = query.transpose(0, 2, 3, 1)
    k = key.transpose(0, 2, 3, 1)
    v = value.transpose(0, 2, 1, 3)

    if not _HAS_RING_KERNEL:
        raise RuntimeError("Ring attention kernel not available")

    attn_output, lse = ring_attention_spmd_fwd(
        q, k, v, prng_key,
        replica_groups=replica_groups,
        num_workers=num_workers,
        softmax_scale=softmax_scale,
        use_causal_mask=causal,
        # Inputs are striped only when causal mask is used
        striped_input=causal,
        mixed_precision=True,
        dropout_p=dropout_rate,
    )

    attn_output = attn_output.transpose(0, 3, 1, 2)
    # Unstripe output only when causal
    if causal:
        attn_output = _unstripe_sequence(attn_output, num_workers)
    return attn_output, (lse, q, k, v, prng_key)


def _ring_backward(
    replica_groups: list,
    num_workers: int,
    causal: bool,
    softmax_scale: float,
    dropout_rate: float,
    res,
    d_attn_output: Tensor,
):
    """Ring attention backward pass."""
    lse, q, k, v, prng_key = res

    # Stripe gradient only when causal
    if causal:
        d_attn_output = _stripe_sequence(d_attn_output, num_workers)
    
    o = d_attn_output.transpose(0, 2, 3, 1)
    dy = d_attn_output.transpose(0, 2, 3, 1)

    if not _HAS_RING_KERNEL:
        raise RuntimeError("Ring attention kernel not available")

    d_query, d_key, d_value = ring_attention_spmd_bwd(
        q, k, v, o, dy, lse, prng_key,
        replica_groups=replica_groups,
        num_workers=num_workers,
        use_causal_mask=causal,
        # Inputs are striped only when causal mask is used
        striped_input=causal,
        # Mixed precision is required to be True in the current impl
        mixed_precision=True,
        dropout_p=dropout_rate,
        softmax_scale=softmax_scale,
    )

    d_query = d_query.transpose(0, 3, 1, 2)
    d_key = d_key.transpose(0, 3, 1, 2)
    d_value = d_value.transpose(0, 3, 1, 2)
    
    # Unstripe gradients only when causal
    if causal:
        d_query = _unstripe_sequence(d_query, num_workers)
        d_key = _unstripe_sequence(d_key, num_workers)
        d_value = _unstripe_sequence(d_value, num_workers)

    return d_query, d_key, d_value, None


ring_attention.defvjp(_ring_forward, _ring_backward)


class NeuronRingAttention(BaseFlashAttention):
    """Wraps the Neuron ring attention kernel for context parallelism."""

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> bool:
        """See `BaseFlashAttention.is_supported`."""
        if not _HAS_RING_KERNEL:
            return self._log_unsupported("ring attention kernel not available.")
        if not super().is_supported(input_batch=input_batch):
            return False
        if self.cfg.dropout_rate != 0.0:
            return self._log_unsupported("dropout is not supported.")

        cp_degree = int(os.getenv("AXLEARN_CP_DEGREE", "1"))
        if cp_degree <= 1:
            return self._log_unsupported("AXLEARN_CP_DEGREE must be > 1.")

        try:
            mesh = jax.interpreters.pxla.thread_resources.env.physical_mesh
            if "seq" not in mesh.axis_names:
                return self._log_unsupported("seq axis not in mesh.")
            seq_size = mesh.shape[mesh.axis_names.index("seq")]
            if seq_size != cp_degree:
                return self._log_unsupported(f"seq axis size {seq_size} != CP_DEGREE {cp_degree}.")
        except (AttributeError, ValueError):
            return self._log_unsupported("unable to access mesh.")
        logging.info("Using %s with context parallelism degree %d.", self.name(), cp_degree)
        return True

    @partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> Tensor:
        """See `BaseFlashAttention.__call__`."""
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        value: Tensor = input_batch["value"]
        bias: BaseAttentionBias = input_batch["bias"]
        prng_key: Tensor = input_batch.get("prng_key", jax.random.key(0))
        key = repeat_kv_heads(query.shape[2], key)
        value = repeat_kv_heads(query.shape[2], value)

        causal, other_biases = split(bias, CausalAttentionBias)
        if other_biases.has_value():
            raise NotImplementedError("Non-causal biases not supported in ring attention")

        mesh = jax.interpreters.pxla.thread_resources.env.physical_mesh
        replica_groups, num_workers = _get_replica_groups_and_workers(mesh, "seq")

        return ring_attention(
            query,
            key,
            value,
            prng_key,
            replica_groups,
            num_workers,
            causal=causal.has_value(),
            softmax_scale=self.cfg.softmax_scale,
            dropout_rate=self.cfg.dropout_rate,
        )
