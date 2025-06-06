# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google/flax:
# Copyright 2024 The Flax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/maxtext:
# Copyright 2024 The MaxText Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/aqt:
# Copyright 2024 The AQT Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Defines QuantizedDotGeneral, the class for hardware accelerated
quantized dot_general operations.

This class should not be directly used. Adopters should inherent from
DenseGeneralBaseLayer instead.
"""
import functools
from enum import Enum
from typing import Optional, Union

import jax
from absl import logging
from aqt.jax.v2.config import DotGeneral, set_context
from jax import numpy as jnp
from jax.lax import DotDimensionNumbers, Precision
from jax.typing import DTypeLike

from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import config_class
from axlearn.common.module import Module
from axlearn.common.param_init import constant_initializer
from axlearn.common.quantized_dot_general.activation_clipping import BaseActivationClippingLayer
from axlearn.common.quantized_dot_general.utils import (
    is_einsum_swapped_operands,
    lhs_activation_aqt_config,
    rhs_activation_aqt_config,
)
from axlearn.common.utils import Tensor

PrecisionLike = Union[None, str, Precision, tuple[str, str], tuple[Precision, Precision]]


class DotGeneralQuantizationType(Enum):
    """Types of hardware accelerated quantization available."""

    # Int 8 quantization. Available on v5e/v5p TPU.
    INT_8 = 0
    # Fp8 quantization: Available on H100 GPU.
    FP_8 = 1


class ClippingChoice(Enum):
    """Which tensor should we apply clipping to."""

    # Input activation into the dot_general op.
    INPUT_ACTIVATION = 0
    # Output of dot_general op.
    OUTPUT_ACTIVATION = 1


# TODO(hanzhi-zhou): use StrEnum once we upgrade to Python 3.11.
class FP8ScaleParams(Enum):
    INPUT_SCALE = "input_scale"
    KERNEL_SCALE = "kernel_scale"
    OUTPUT_GRADE_SCALE = "output_grad_scale"


class FP8AmaxHistoryParams(Enum):
    INPUT_AMAX_HISTORY = "input_amax_history"
    KERNEL_AMAX_HISTORY = "kernel_amax_history"
    OUTPUT_GRAD_AMAX_HISTORY = "output_grad_amax_history"


def get_all_fp8_param_names():
    return [x.value for x in FP8ScaleParams] + [x.value for x in FP8AmaxHistoryParams]


class QuantizedDotGeneral(BaseLayer):
    """Hardware accelerated quantized dot general layer.

    This layer offers hardware accelerated lower width dot_general and einsum operations,
    providing training speed up while maintaining relatively low quality degradation.

    Users should not use this layer directly. Access its functionality through
    `axlearn.common.dense_general.DenseGeneralBaseLayer` instead.

    Available quantization methods include:
        1. int8 quantization: Available on v5e/v5p TPU.
        2. fp8 quantization: Available on H100 GPU.

    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures QuantizedDotGeneral."""

        # Type of quantization.
        quantization_type: Optional[DotGeneralQuantizationType] = None
        # Activation clipping method. Only necessary for int8 quantization.
        activation_clipping: Optional[BaseActivationClippingLayer.Config] = None
        # Which tensor to apply clipping to. Defaults to input activation.
        clipping_choice: ClippingChoice = ClippingChoice.INPUT_ACTIVATION
        # The length of the absolute maximum (amax) history for fp8 delayed scaling in which the
        # scaling factor is computed based on the amax stats from prior iterations. If set to
        # 0, None or negative, in-batch scaling will be used, and the scaling factor is computed
        # from the activations of the current iteration. For more details, see
        # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
        fp8_amax_history_length: Optional[int] = None

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        logging.info("QuantizedDotGeneral init")
        cfg = self.config
        # Init quantization according to type.
        if cfg.quantization_type == DotGeneralQuantizationType.INT_8:
            # TODO(jiarui): Is there a way to identify if we are running on v5?
            if jax.default_backend() != "tpu":
                logging.warning(
                    "Hardware accelerated Int8 quantization is only available on "
                    "v5litepod/v5p TPU, found %s. When running on incompatible backend, "
                    "quantization still happens, but step time will regress comparing to "
                    "bf16.",
                    jax.default_backend(),
                )
            logging.info("Int8 enabled")
            # Int8 quantization does not require additional parameters,
            # but does require prng_key at call time
            # to perform stochastic rounding for rhs backward pass.
            # Because of this we don't need to add_child
            # for anything, we just need to init an aqt_dot_general function
            # with recommended configs.
            # Dot general with default config.
            self.lhs_act_dot_general: DotGeneral = lhs_activation_aqt_config()
            # Dot general with mirrored config where lhs and rhs are swapped.
            self.rhs_act_dot_general: DotGeneral = rhs_activation_aqt_config()
            if cfg.fp8_amax_history_length is not None:
                raise ValueError(
                    "fp8_amax_history_length should not be specified when using "
                    "Int8 quantization."
                )
        elif cfg.quantization_type == DotGeneralQuantizationType.FP_8:
            # TODO(jiarui): Is there a way to identify if we are running on H100?
            if jax.default_backend() != "gpu":
                raise NotImplementedError("Fp8 quantization is only available on H100 GPU")
        elif cfg.quantization_type is not None:
            raise KeyError(
                f"Unrecognized quantization type {cfg.quantization_type}. "
                f"Available types {list(DotGeneralQuantizationType)}"
            )

        # Init activation clipping.
        if cfg.activation_clipping is not None:
            self._add_child("activation_clipping", cfg.activation_clipping)

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        specs = super()._create_layer_parameter_specs()
        cfg: QuantizedDotGeneral.Config = self.config
        if cfg.quantization_type == DotGeneralQuantizationType.FP_8:
            scale_spec = ParameterSpec(
                shape=(),
                dtype=jnp.float32,
                mesh_axes=(),
                initializer=constant_initializer(1.0),
                weight_decay_scale=0,
            )
            specs.update({k.value: scale_spec for k in FP8ScaleParams})
            if cfg.fp8_amax_history_length is not None and cfg.fp8_amax_history_length > 0:
                amax_spec = ParameterSpec(
                    shape=[cfg.fp8_amax_history_length],
                    dtype=jnp.float32,
                    mesh_axes=(None,),
                    initializer=constant_initializer(0.0),
                    weight_decay_scale=0,
                )
                specs.update({k.value: amax_spec for k in FP8AmaxHistoryParams})
        return specs

    def _dot_general_maybe_quantized(
        self,
        lhs: Tensor,
        rhs: Tensor,
        dimension_numbers: DotDimensionNumbers,
        precision: PrecisionLike = None,
        preferred_element_type: Optional[DTypeLike] = None,
        prng_key: Optional[Tensor] = None,
        lhs_is_activation: bool = True,
    ) -> Tensor:
        """Utilize hardware accelerated quantized dot_general depending on config and hardware.

        See Also: https://github.com/google/maxtext/blob/main/MaxText/layers/quantizations.py

        Args:
            lhs: Left hand side tensor of dot_general.
            rhs: Right hand side tensor of dot_general.
            dimension_numbers: a tuple of tuples of sequences of ints of the form
                ``((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))``
            precision: Optional. Either ``None``, which means the default precision for
                the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
                ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
                :class:`~jax.lax.Precision` enums indicating precision of ``lhs``` and ``rhs``.
            preferred_element_type: Optional. Either ``None``, which means the default
                accumulation type for the input types, or a datatype, indicating to
                accumulate results to and return a result with that datatype.
            prng_key: Optional. PRNG key used for int8 stochastic rounding. When it is None,
                default to self.prng_key.
            lhs_is_activation: A boolean indicating if lhs represents activation or not.
                Only necessary for int8 quantization.
                Int8 quantization requires stochastic rounding on activation gradient.
                Because of this we need to know which operand represents activation,
                and call corresponding quantized dot_general accordingly.
        Returns:
            An array whose first dimensions are the (shared) batch dimensions, followed by
            the ``lhs`` non-contracting/non-batch dimensions, and finally the ``rhs``
            non-contracting/non-batch dimensions.
        """
        cfg: QuantizedDotGeneral.Config = self.config
        # Choose between quantization options.
        # Assumes that config and hardware pairing is already validated in __init__.
        if cfg.quantization_type is None:
            # If no quantization is needed, default to jax.lax.dot_general.
            return jax.lax.dot_general(
                lhs, rhs, dimension_numbers, precision, preferred_element_type
            )
        elif cfg.quantization_type == DotGeneralQuantizationType.INT_8:
            # Provide prng_key and call self.aqt_dot_general.
            if lhs_is_activation:
                fn: DotGeneral = self.lhs_act_dot_general
            else:
                fn = self.rhs_act_dot_general
            # Pass in prng_key for stochastic rounding
            set_context(
                cfg=fn, key=prng_key if prng_key is not None else self.prng_key, train_step=None
            )
            return fn(
                lhs,
                rhs,
                dimension_numbers=dimension_numbers,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )
        elif cfg.quantization_type == DotGeneralQuantizationType.FP_8:
            return self._fp8_dot(
                lhs,
                rhs,
                dimension_numbers=dimension_numbers,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )
        else:
            raise KeyError(
                f"Unrecognized quantization type {cfg.quantization_type}. "
                f"Available types {list(DotGeneralQuantizationType)}"
            )

    def _fp8_dot(self, lhs, rhs, dimension_numbers, precision, preferred_element_type):
        # Use delayed import to avoid global dependency on flax.linen.
        # pylint: disable-next=import-outside-toplevel
        from axlearn.common.quantized_dot_general import fp8_ops

        is_delayed = self.config.fp8_amax_history_length > 0
        logging.log_first_n(
            logging.INFO, "Using fp8 %s scaling.", 1, "delayed" if is_delayed else "in batch"
        )
        scale_params = [self.parameters[x.value] for x in FP8ScaleParams]
        history_params = (
            [self.parameters[x.value] for x in FP8AmaxHistoryParams] if is_delayed else [None] * 3
        )
        fn = fp8_ops.q_dot_q
        if (
            len(scale_params[0].shape) > 0
            and lhs.shape[0] == rhs.shape[0]
            and lhs.shape[0] == scale_params[0].shape[0]
        ):
            # For batched matmul of the form (B, M, K) @ (B, K, N), use vmap so that each
            # batch of the lhs/rhs can have different scaling factors.
            temp = [None] * 4
            ((temp[0], temp[1]), (temp[2], temp[3])) = dimension_numbers
            # Filter out dim 0 from lhs/rhs batch dimensions.
            for i in range(2, 4):
                temp[i] = tuple(filter(lambda x: x != 0, temp[i]))
            # Reduce all dim numbers by 1, since vmap removes the outer dimension.
            for i in range(4):
                temp[i] = tuple(map(lambda x: x - 1, temp[i]))
            dimension_numbers = ((temp[0], temp[1]), (temp[2], temp[3]))
            fn = jax.vmap(
                fn, in_axes=[0] * 5 + ([0] * 3 if is_delayed else [None] * 3) + [None] * 3
            )
        return fn(
            lhs,
            rhs,
            *scale_params,
            *history_params,
            dimension_numbers,
            precision,
            preferred_element_type,
        )

    def einsum_maybe_quantized(self, subscripts, *, activation: Tensor, kernel: Tensor) -> Tensor:
        """jnp.einsum which uses hardware accelerated quantization if applicable.

        See docstring for jax.numpy.einsum.

        Note that you should only use 2 operands with this function, with the lhs
        operand being activation and the rhs one being model weight.

        Args:
            subscripts: Specifies the subscripts for summation as comma separated
                list of subscript labels.
                An implicit (classical Einstein summation) calculation is
                performed unless the explicit indicator ‘->’ is included
                as well as subscript labels of the precise output form.
            activation: Activation tensor. AQT einsum requires activation in lhs.
            kernel: Kernel tensor. AQT einsum requires kernel in rhs.

        Returns:
            Output of einsum.
        """
        cfg = self.config
        is_swapped: bool = is_einsum_swapped_operands(subscripts, activation, kernel)
        # Apply clipping if applicable.
        # Adding summaries within the _dot_general function will cause a side effect,
        # So we had to do this outside of dot_general_maybe_quantized. Not sure
        # if there's a better way to fix this.
        # Apply clipping on input activation.
        if (
            "activation_clipping" in self.children
            and cfg.clipping_choice == ClippingChoice.INPUT_ACTIVATION
        ):
            activation = self.activation_clipping(activation)
        output = jnp.einsum(
            subscripts,
            activation,
            kernel,
            _dot_general=functools.partial(
                self._dot_general_maybe_quantized,
                prng_key=self.prng_key,
                lhs_is_activation=not is_swapped,
            ),
        )
        # Apply clipping on output.
        if (
            "activation_clipping" in self.children
            and cfg.clipping_choice == ClippingChoice.OUTPUT_ACTIVATION
        ):
            output = self.activation_clipping(output)
        return output


class DenseGeneralBaseLayer(BaseLayer):
    """Base Layer for all Linear transformations.

    Directly utilizes dot_general operation. Optionally applies hardware accelerated
    quantized dot_general depending on config.

    Adopting users intending to use QuantizedDotGeneral in SomeLayer will need to
    follow these steps:

    1. Inherent from `DenseGeneralBaseLayer`.
    2. Call `self.einsum_maybe_quantized` instead of `jax.numpy.einsum`.
    3. Recursively populate `quantized_dot_general` configs with
        `axlearn.common.layers.set_quantized_dot_general_recursively`.
    """

    @config_class
    class Config(BaseLayer.Config):
        # QuantizedDotGeneral config. Replace jax.lax.dot_general with
        # dot_general_maybe_quantized from this layer to achieve
        # hardware accelerated quantized dot_general.
        # TODO(jiarui): Switch to ConfigOr[Protocol] which defines the implementation of
        #  dot_general(subscript, activation, kernel) (which can be a layer) to generalize the API.
        quantized_dot_general: Optional[QuantizedDotGeneral.Config] = None

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.quantized_dot_general is not None:
            self._add_child("quantized_dot_general", cfg.quantized_dot_general)

    def einsum_maybe_quantized(
        self, subscript: str, *, activation: Tensor, kernel: Tensor
    ) -> Tensor:
        """Computes einsum with `layer.quantized_dot_general` if available.

        Args:
            subscript: Einsum subscript.
            activation: Activation tensor. AQT einsum requires activation in lhs.
            kernel: Kernel tensor. AQT einsum requires kernel in rhs.

        Returns:
            Einsum result.
        """
        if "quantized_dot_general" in self.children:
            return self.quantized_dot_general.einsum_maybe_quantized(
                subscript, activation=activation, kernel=kernel
            )
        else:
            return jnp.einsum(subscript, activation, kernel)


def set_quantized_dot_general_recursively(
    cfg: BaseLayer.Config,
    quantized_dot_general: Optional[QuantizedDotGeneral.Config],
    set_only_if_none: bool = False,
):
    """Sets QuantizedDotGeneral.Config recursively.

    QuantizedDotGeneral is only used in DenseGeneralBaseLayer and its
    subclasses. This function identifies these configs, and sets
    QuantizedDotGeneral.Config accordingly.

    Args:
        cfg: The root config under which to set. QuantizedDotGeneral.Config.
        quantized_dot_general: QuantizedDotGeneral Config.
        set_only_if_none: Override only when original value is None.
    """

    def visit_fn(_, value):
        if isinstance(value, DenseGeneralBaseLayer.Config):
            if not set_only_if_none or value.quantized_dot_general is None:
                value.quantized_dot_general = quantized_dot_general

    def enter_fn(_, value, default_kv):
        return None if isinstance(value, DenseGeneralBaseLayer.Config) else default_kv

    cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)
