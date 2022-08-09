#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
References

.. [burt2020svgp]
    David R. Burt and Carl Edward Rasmussen and Mark van der Wilk,
    Convergence of Sparse Variational Inference in Gaussian Process Regression,
    Journal of Machine Learning Research, 2020,
    http://jmlr.org/papers/v21/19-1015.html.

.. [chen2018dpp]
    Laming Chen and Guoxin Zhang and Hanning Zhou, Fast greedy MAP inference
    for determinantal point process to improve recommendation diversity,
    Proceedings of the 32nd International Conference on Neural Information
    Processing Systems, 2018, https://arxiv.org/abs/1709.05135.

.. [hensman2013svgp]
    James Hensman and Nicolo Fusi and Neil D. Lawrence, Gaussian Processes
    for Big Data, Proceedings of the 29th Conference on Uncertainty in
    Artificial Intelligence, 2013, https://arxiv.org/abs/1309.6835.

"""

# from __future__ import annotations

from typing import Optional, Type, Union

import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling import MCSampler
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.lazy import LazyTensor
from gpytorch.likelihoods import (
    GaussianLikelihood,
    Likelihood,
    MultitaskGaussianLikelihood,
)
from gpytorch.means import ConstantMean, Mean
from gpytorch.models import ApproximateGP
from gpytorch.priors import GammaPrior
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
    _VariationalDistribution,
    _VariationalStrategy,
)
from torch import Tensor


MIN_INFERRED_NOISE_LEVEL = 1e-4


def _pivoted_cholesky_init(
    train_inputs: Tensor,
    kernel_matrix: Union[Tensor, LazyTensor],
    max_length: int,
    epsilon: float = 1e-6,
) -> Tensor:
    r"""
    A pivoted cholesky initialization method for the inducing points,
    originally proposed in [burt2020svgp]_ with the algorithm itself coming from
    [chen2018dpp]_. Code is a PyTorch version from [chen2018dpp]_, copied from
    https://github.com/laming-chen/fast-map-dpp/blob/master/dpp.py.

    Args:
        train_inputs [Tensor]: training inputs
        kernel_matrix [Tensor or Lazy Tensor]: kernel matrix on the training
            inputs
        max_length [int]: number of inducing points to initialize
        epsilon [float]: numerical jitter for stability.
    """

    # this is numerically equivalent to iteratively performing a pivoted cholesky
    # while storing the diagonal pivots at each iteration
    # TODO: use gpytorch's pivoted cholesky instead once that gets an exposed list
    # TODO: this probably won't work in batch mode.

    item_size = kernel_matrix.shape[-2]
    cis = torch.zeros((max_length, item_size))
    di2s = kernel_matrix.diag()
    selected_items = []
    selected_item = torch.argmax(di2s)
    selected_items.append(selected_item)

    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel_matrix[..., selected_item, :]
        eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s = di2s - eis.pow(2.0)
        di2s[selected_item] = -(torch.tensor(float("inf")))
        selected_item = torch.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)

    ind_points = train_inputs[torch.stack(selected_items)]

    return ind_points


class ApproximateGPyTorchModel(GPyTorchModel):
    def __init__(
        self,
        model: Optional[ApproximateGP] = None,
        likelihood: Optional[Likelihood] = None,
        num_outputs: int = 1,
        *args,
        **kwargs,
    ):
        r"""
        Botorch wrapper class for various (variational) approximate GP models in
        gpytorch. This can either include stochastic variational GPs (SVGPs) or
        variational implementations of weight space approximate GPs.

        model: instance of gpytorch.approximate GP models
        likelihood: instance of gpytorch likelihood
        num_outputs: number of outputs expected for the GP model
        """

        super().__init__()

        if model is None:
            model = _SingleTaskVariationalGP(num_outputs=num_outputs, **kwargs)

        if likelihood is None:
            if num_outputs == 1:
                likelihood = GaussianLikelihood()
            else:
                likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)

        self.model = model
        self.likelihood = likelihood
        self._desired_num_outputs = num_outputs

    @property
    def num_outputs(self):
        return self._desired_num_outputs

    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode
        self.model.eval()
        self.likelihood.eval()

        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)

        dist = self.model(X)
        if observation_noise:
            dist = self.likelihood(dist, *args, **kwargs)

        return GPyTorchPosterior(mvn=dist)

    def forward(self, X, *args, **kwargs) -> MultivariateNormal:
        X = self.transform_inputs(X)
        return self.model(X)

    def fantasize(self, X, sampler=MCSampler, observation_noise=True, *args, **kwargs):
        raise NotImplementedError(
            "Fantasization of approximate GPs has not been implemented yet."
        )


class _SingleTaskVariationalGP(ApproximateGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        num_outputs: int = 1,
        learn_inducing_points=True,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
        init_method: str = "pivchol",
    ) -> None:
        r"""
        Base class wrapper for a stochastic variational Gaussian Process (SVGP)
        model [hensman2013svgp]_.

        Args:
            train_X: training inputs (note that this doesn't have to be all of the
                training targets)
            train_Y: training targets (optional)
            num_outputs: number of output responses per input (default: 1).
            covar_module: kernel function
            mean_module: mean of GP model
            variational_distribution: type of variational distribution to use
                (default: CholeskyVariationalDistribution), the properties of the
                variational distribution will encourage scalability or ease of
                optimization.
            variational_strategy: type of variational strategy to use (default:
                VariationalStrategy). The default setting uses "whitening" of the
                variational distribution to make training easier.
            inducing_points: if inducing points should be pre-specified.
            init_method: initialization method for the inducing points
                (default: pivoted cholesky).
        """
        # we use the model subclass wrapper to deal with input / outcome transforms

        batch_shape = train_X.shape[:-2]
        if num_outputs > 1:
            batch_shape = torch.Size((num_outputs,)) + batch_shape
        self._aug_batch_shape = batch_shape

        if mean_module is None:
            mean_module = ConstantMean(batch_shape=self._aug_batch_shape)

        if covar_module is None:
            covar_module = ScaleKernel(
                base_kernel=MaternKernel(
                    nu=2.5,
                    ard_num_dims=train_X.shape[-1],
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            self._subset_batch_dict = {
                "mean_module.constant": -2,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }

        # initialize inducing points with a pivoted cholesky init if they are not given
        if type(inducing_points) is not Tensor:
            if inducing_points is None:
                # number of inducing points is 25% the number of data points
                # as a heuristic
                inducing_points = int(0.25 * train_X.shape[-2])

            if init_method == "pivchol":
                with torch.no_grad():
                    train_train_kernel = covar_module(train_X)
                    if train_train_kernel.ndimension() > 2:
                        train_train_kernel = train_train_kernel.evaluate_kernel()[0]
                        need_to_repeat_over_batch = True
                    else:
                        need_to_repeat_over_batch = False

                    inducing_points = _pivoted_cholesky_init(
                        train_X, train_train_kernel, max_length=inducing_points
                    )

                    if need_to_repeat_over_batch:
                        inducing_points = inducing_points.unsqueeze(0)
                        inducing_points = inducing_points.repeat(
                            *batch_shape, *([1] * (inducing_points.ndimension() - 1))
                        )
            else:
                raise (
                    AssertionError(
                        "Only pivoted cholesky initialization has been implemented."
                    )
                )

        if variational_distribution is None:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.shape[-2], batch_shape=batch_shape
            )

        variational_strategy = variational_strategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_points,
        )

        # wrap variational models in independent multi-task variational strategy
        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                base_variational_strategy=variational_strategy,
                num_tasks=num_outputs,
                task_dim=-1,
            )
        super().__init__(variational_strategy=variational_strategy)

        self._aug_batch_shape = batch_shape

        self.mean_module = mean_module

        self.covar_module = covar_module

    def forward(self, x) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_dist = MultivariateNormal(mean_x, covar_x)
        return latent_dist


class SingleTaskVariationalGP(ApproximateGPyTorchModel):
    r"""A single-task variational GP model following [hensman2013svgp]_.

    A single-task variational GP using relatively strong priors on the Kernel
    hyperparameters, which work best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance).

    This model works in batch mode (each batch having its own hyperparameters).
    When the training observations include multiple outputs, this model will use
    batching to model outputs independently.

    Use this model if you have a lot of data or if your responses are non-Gaussian.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        num_outputs: int = 1,
        learn_inducing_points: bool = True,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
        init_method: str = "pivchol",
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""
        A single task stochastic variational Gaussian process model (SVGP) as described
        by [hensman2013svgp]_. We use pivoted cholesky initialization [burt2020svgp]_ to
        initialize the inducing points of the model.

        Args:
            train_X: training inputs
            likelihood: likelihood of the responses (generally will be
                GaussianLikelihood)
            num_outputs: number of output responses per input (default: 1).
            covar_module: kernel function
            mean_module: mean of GP model
            variational_distribution: type of variational distribution to use (default:
                CholeskyVariationalDistribution), the properties of the variational
                distribution will enhance scalability or ease of optimization.
            variational_strategy: type of variational strategy to use (default:
                VariationalStrategy). The default setting uses "whitening" of the
                variational distribution to make training easier.
            inducing_points: if inducing points should be pre-specified.
            init_method: initialization method for the inducing points
                (default: pivoted cholesky).
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if train_Y is not None:
            if outcome_transform is not None:
                train_Y, _ = outcome_transform(train_Y)
            self._validate_tensor_args(X=transformed_X, Y=train_Y)
            validate_input_scaling(train_X=transformed_X, train_Y=train_Y)
            if train_Y.shape[-1] != num_outputs:
                num_outputs = train_Y.shape[-1]

        self._num_outputs = num_outputs
        self._input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = self._input_batch_shape
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        self._aug_batch_shape = aug_batch_shape

        if likelihood is None:
            if num_outputs == 1:
                noise_prior = GammaPrior(1.1, 0.05)
                noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
                likelihood = GaussianLikelihood(
                    noise_prior=noise_prior,
                    batch_shape=self._aug_batch_shape,
                    noise_constraint=GreaterThan(
                        MIN_INFERRED_NOISE_LEVEL,
                        transform=None,
                        initial_value=noise_prior_mode,
                    ),
                )
            else:
                likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)
        else:
            self._is_custom_likelihood = True

        model = _SingleTaskVariationalGP(
            train_X=transformed_X,
            num_outputs=num_outputs,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            init_method=init_method,
        )

        super().__init__(model=model, likelihood=likelihood)

        # for model fitting utilities
        # TODO: make this a flag?
        self.training_inputs = [train_X]
        self.train_targets = train_Y