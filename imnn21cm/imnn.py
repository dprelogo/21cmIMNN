"""
Information maximizing neural network: unsupervised neural summary.
The algorithm consists in maximizing Fisher information in the summary space,
with respect to the input parameters.
"""

import pickle
import cloudpickle
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map

from imnn21cm.tree_util import tree_stack, replicate_array, strip_array
from imnn21cm._base import IMNNBase

# from flax.core import freeze, unfreeze
from flax import serialization
import optax

import numpy as np
import h5py


class IMNN(IMNNBase):
    """Full IMNN. Training is done so that one epoch consists of computing
    summaries over the whole training set, computing loss and updating gradients.
    """

    def __init__(self, *args, **kwargs):
        super(IMNN, self).__init__(*args, **kwargs)

    def _initialize_model(self, model, model_kwargs=None):
        self.model = model
        # if "batch_stats" is sent to model.apply,
        # and in the case such parameters exist, it will return them as a state
        # otherwise it will return an empty state
        if model_kwargs is None:
            self.model_kwargs = {"mutable": ["batch_stats"]}
        else:
            self.model_kwargs = model_kwargs

        variables = self.model.init(
            self._get_key(), jnp.ones((1,) + self.input_dim), use_running_average=False
        )
        self.model_state, self.model_params = variables.pop("params")
        del variables

    def _initialize_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer_state = self.optimizer.init(self.model_params)

    def save(self, filename="imnn", compression=None):
        compression = (
            {
                "compression": "gzip",
                "compression_opts": 7,
                "shuffle": True,
            }
            if compression is None
            else compression
        )

        with h5py.File(filename + "_history.h5", "w") as f:
            f.create_dataset(
                "loss",
                data=np.array(self.history["loss"], dtype=np.float32),
                **compression,
            )
            f.create_dataset(
                "loss_val",
                data=np.array(self.history["loss_val"], dtype=np.float32),
                **compression,
            )

            f_stats = f.create_group("stats")
            for k, v in self.history["stats"].items():
                f_stats.create_dataset(
                    k, data=np.array(v, dtype=np.float32), **compression
                )
            f_stats_val = f.create_group("stats_val")
            for k, v in self.history["stats_val"].items():
                f_stats_val.create_dataset(
                    k, data=np.array(v, dtype=np.float32), **compression
                )

        with open(filename + "_params.bin", "wb") as f:
            f.write(serialization.to_bytes(self.model_params))
        with open(filename + "_state.bin", "wb") as f:
            f.write(serialization.to_bytes(self.model_state))
        with open(filename + "_optimizer.pkl", "wb") as f:
            cloudpickle.dump((self.optimizer, self.optimizer_state), f)

    def load(self, filename="imnn", load_optimizer=True, load_history=True):
        if load_history:
            with h5py.File(filename + "_history.h5", "r") as f:
                self.history["loss"] = list(f["loss"][()])
                self.history["loss_val"] = list(f["loss_val"][()])
                f_stats = f["stats"]
                f_stats_val = f["stats_val"]
                self.history["stats"] = {k: list(v[()]) for k, v in f_stats.items()}
                self.history["stats_val"] = {
                    k: list(v[()]) for k, v in f_stats_val.items()
                }

        with open(filename + "_params.bin", "rb") as f:
            self.model_params = serialization.from_bytes(
                self.model_params, f.read()
            )  # passing model_params just to confirm the model shape
        with open(filename + "_state.bin", "rb") as f:
            self.model_state = serialization.from_bytes(
                self.model_state, f.read()
            )  # here as well
        if load_optimizer:
            with open(filename + "_optimizer.pkl", "rb") as f:
                self.optimizer, self.optimizer_state = pickle.load(f)

    def loss(
        self, params, state, fiducial, derivatives, noise=None, return_summaries=False
    ):
        """Computing the loss of the whole dataset, for the current values
        of `param` and `state`.

        Args:
            params: parameters of the model
            state: state of the model (e.g. batch norm moving averages etc.)
            fiducial: fiducial dataset
            derivatives: derivative dataset
            noise: additive telescope noise. `None` value assumes the data is
                validation, for which the noise is already added.
            return_summaries: If `True`, returns all summaries
                 of fiducial and derivatives, as a part of `aux`.

        Returns:
            `(L, (aux, new_state))` in the case `noise is not None`,
            where `L` is the loss, `aux` auxiliary data logged in history and
            `new_state` updated model state.

            `(L, aux)` pair in the case `noise is None`.
        """

        @partial(jax.pmap, devices=self.gpus, backend="gpu")
        def collect_predictions_fid(x):
            # this function is called only for validation
            # therefore we need to use_running_average,
            # and we don't need to define mutable variables
            return self.model.apply(
                {"params": params, **state}, x, use_running_average=True
            )

        @partial(jax.pmap, devices=self.gpus, backend="gpu")
        def collect_predictions_fid_noise(x, noise):
            # this function is called for training
            # therefore we calculate running average,
            # and we need to define mutable variables
            return self.model.apply(
                {"params": params, **state},
                x + noise,
                **self.model_kwargs,
                use_running_average=False,
            )

        @partial(jax.pmap, devices=self.gpus, backend="gpu")
        @partial(jax.vmap, in_axes=(1,), out_axes=1)
        @partial(jax.vmap, in_axes=(1,), out_axes=1)
        def collect_predictions_der(x):
            return self.model.apply(
                {"params": params, **state}, x, use_running_average=True
            )

        @partial(jax.pmap, devices=self.gpus, backend="gpu")
        @partial(jax.vmap, in_axes=(1, None), out_axes=1)
        @partial(jax.vmap, in_axes=(1, None), out_axes=1)
        def collect_predictions_der_noise(x, noise):
            return self.model.apply(
                {"params": params, **state},
                x + noise,
                **self.model_kwargs,
                use_running_average=False,
            )

        if noise is None:
            # for validation the function doesn't return state, so I can just stack results
            summaries_fiducial = jnp.stack(
                [collect_predictions_fid(x) for x in fiducial]
            )
            summaries_derivatives = jnp.stack(
                [collect_predictions_der(x) for x in derivatives]
            )
        else:
            # for training, the function returns the state
            # thus I need to stack only results, and average states
            noise_fiducial, noise_derivatives = noise
            fiducial_result = [
                collect_predictions_fid_noise(x, n)
                for x, n in zip(fiducial, noise_fiducial)
            ]
            summaries_fiducial = jnp.stack([x for x, y in fiducial_result])
            derivatives_result = [
                collect_predictions_der_noise(x, n)
                for x, n in zip(derivatives, noise_derivatives)
            ]
            summaries_derivatives = jnp.stack([x for x, y in derivatives_result])

            fiducial_state = tree_stack([y for x, y in fiducial_result])
            derivatives_state = tree_stack([y for x, y in derivatives_result])

            # exact averaging of the state
            new_state = tree_map(
                partial(self.average_xy, x_axis=(0, 1), y_axis=(0, 1, 3, 4)),
                fiducial_state,
                derivatives_state,
            )

        summaries_fiducial = summaries_fiducial.reshape(
            (self.n_sims // 2, self.n_summaries)
        )
        summaries_derivatives = summaries_derivatives.reshape(
            (self.n_derivatives // 2, 2, self.n_params, self.n_summaries)
        )
        if return_summaries:
            summaries_aux = {
                "summaries_fiducial": summaries_fiducial.copy(),
                "summaries_derivatives": summaries_derivatives.copy(),
            }
        δθ = jnp.reshape(self.δθ, (1, self.n_params, 1))
        summaries_derivatives = (
            summaries_derivatives[:, 1, ...] - summaries_derivatives[:, 0, ...]
        ) / δθ

        if noise is None:
            F, C, invC, dμ_dθ = self.F_statistics_full(
                summaries_fiducial, summaries_derivatives
            )
        else:
            F, C, invC, dμ_dθ = self.F_statistics_train(
                summaries_fiducial, summaries_derivatives
            )

        lndetF = self.slogdet(F)
        Λ2 = self.regularization(C, invC)
        r = self.regularization_strength(Λ2)

        L = (
            -lndetF
            + r * Λ2 * self.C_regularization_strength
            + self.l2_norm(params) * self.weight_regularization_strength
        )

        aux = {}
        for k, v in zip(["F", "C", "Cinv", "r", "Λ2"], [F, C, invC, r, Λ2]):
            aux[k] = jnp.array(v, dtype=jnp.float32)
        if return_summaries:
            for k, v in summaries_aux.items():
                aux[k] = v
        if noise is None:
            return L, aux
        else:
            return L, (aux, new_state)

    def _update_optimizer(
        self,
        i,
        fiducial,
        fiducial_val,
        noise_fiducial,
        derivatives,
        derivatives_val,
        noise_derivatives,
    ):
        #         params = self.optimizer["get_params"](self.optimizer["state"])
        (L, (stats, new_state)), gradient = jax.value_and_grad(self.loss, has_aux=True)(
            self.model_params,
            self.model_state,
            fiducial,
            derivatives,
            (noise_fiducial, noise_derivatives),
        )
        self.history["loss"].append(np.array(L, dtype=np.float32))
        for k, v in stats.items():
            self.history["stats"][k].append(np.array(v, dtype=np.float32))

        if not jnp.sum(jnp.isnan(L)):
            updates, self.optimizer_state = self.optimizer.update(
                gradient, self.optimizer_state
            )
            self.model_params = optax.apply_updates(self.model_params, updates)
            self.model_state = new_state

        L_val, stats_val = self.loss(
            self.model_params, self.model_state, fiducial_val, derivatives_val
        )
        self.history["loss_val"].append(np.array(L_val, dtype=np.float32))
        for k, v in stats_val.items():
            self.history["stats_val"][k].append(np.array(v, dtype=np.float32))

    def train(
        self,
        epochs=10,
        verbose=True,
        verbose_frequency=1,
        save=False,
        save_frequency=5,
        save_filename="imnn_{}",
    ):
        N_epochs_trained = len(self.history["loss"])
        for ep in range(N_epochs_trained, N_epochs_trained + epochs):
            if self.prefetch_data is True:
                finite_iterators = (
                    self._finite_it(x)
                    for x in (
                        self.fiducial,
                        self.fiducial_val,
                        self.noise_fiducial,
                        self.derivatives,
                        self.derivatives_val,
                        self.noise_derivatives,
                    )
                )
                self._update_optimizer(ep, *finite_iterators)
            else:
                noise_fiducial, noise_derivatives = self._permute(
                    self.noise_fiducial, self.noise_derivatives
                )
                fiducial, derivatives = self._permute(self.fiducial, self.derivatives)
                self._update_optimizer(
                    ep,
                    fiducial,
                    self.fiducial_val,
                    noise_fiducial,
                    derivatives,
                    self.derivatives_val,
                    noise_derivatives,
                )

            if verbose and (ep + 1) % verbose_frequency == 0:
                self.print_current_stats()
            if save and (ep + 1) % save_frequency == 0:
                self.save(filename=save_filename.format(ep + 1), compression={})

    def predict(
        self,
        data,
        batch_size=1,
    ):
        """Predicting the data with the model

        Args:
            data: data to predict, of shape `(N,) + input_dim`.
            batch_size: batch size for the prediction. Number of samples should
                be divisible by `batch_size * ndevices`
        """
        data_shape = data.shape
        N = data_shape[0]
        if data_shape[1:] != self.input_dim:
            raise ValueError(f"Data is not of shape `(N,)` + {self.input_dim}")
        if N % (batch_size * len(self.gpus)) != 0:
            raise ValueError(
                f"Number of samples is not divisible by `batch_size * ndevices`"
            )

        @partial(jax.pmap, devices=self.gpus, backend="gpu")
        def collect(x):
            return self.model.apply(
                {"params": self.model_params, **self.model_state},
                x,
                use_running_average=True,
            )

        data = data.reshape(
            (N // batch_size // len(self.gpus), len(self.gpus), batch_size)
            + self.input_dim
        )

        summaries = jnp.stack([collect(x) for x in data])
        return summaries.reshape((N, self.n_summaries))


# check dimensions for model_state
class IMNN_SGD(IMNN):
    """Stochastic gradient descent IMNN.
    During training, the loss function and gradients are calculated per batch,
    on every GPU speparately and then averaged.
    For validation, a full aggregated Fisher is calculated.
    """

    def __init__(self, *args, **kwargs):
        super(IMNN_SGD, self).__init__(*args, **kwargs)

    @partial(jax.jit, backend="gpu", static_argnums=0)
    def train_loss(
        self, params, state, fiducial, derivatives, noise_fiducial, noise_derivatives
    ):
        def collect_predictions_fid(x, noise):
            return self.model.apply(
                {"params": params, **state},
                x + noise,
                **self.model_kwargs,
                use_running_average=False,
            )

        @partial(jax.vmap, in_axes=(1, None), out_axes=1)
        @partial(jax.vmap, in_axes=(1, None), out_axes=1)
        def collect_predictions_der(x, noise):
            return self.model.apply(
                {"params": params, **state},
                x + noise,
                **self.model_kwargs,
                use_running_average=False,
            )

        summaries_fiducial, fiducial_state = collect_predictions_fid(
            fiducial, noise_fiducial
        )
        summaries_derivatives, derivatives_state = collect_predictions_der(
            derivatives, noise_derivatives
        )

        δθ = jnp.reshape(self.δθ, (1, self.n_params, 1))
        summaries_derivatives = (
            summaries_derivatives[:, 1, ...] - summaries_derivatives[:, 0, ...]
        ) / δθ

        F, C, invC, dμ_dθ = self.F_statistics_train(
            summaries_fiducial, summaries_derivatives
        )
        lndetF = self.slogdet(F)
        Λ2 = self.regularization(C, invC)
        r = self.regularization_strength(Λ2)
        L = (
            -lndetF
            + r * Λ2
            + self.l2_norm(params) * self.weight_regularization_strength
        )

        aux = {}
        for k, v in zip(["F", "C", "Cinv", "r", "Λ2"], [F, C, invC, r, Λ2]):
            #             aux[k] = jnp.array(v, dtype = jnp.float32)
            aux[k] = v
        return L, (aux, fiducial_state, derivatives_state)

    def _update_optimizer(
        self, i, fiducial, noise_fiducial, derivatives, noise_derivatives
    ):
        #         params = self.optimizer["get_params"](self.optimizer["state"])

        @partial(jax.pmap, axis_name="i", devices=jax.devices("gpu"), backend="gpu")
        def train_step(fiducial, derivatives, noise_fiducial, noise_derivatives):
            (
                L,
                (stats, fiducial_state, derivatives_state),
            ), gradient = jax.value_and_grad(self.train_loss, has_aux=True)(
                self.model_params,
                self.model_state,
                fiducial,
                derivatives,
                noise_fiducial,
                noise_derivatives,
            )
            gradient = jax.lax.pmean(gradient, "i")
            L = jax.lax.pmean(L, "i")
            for v in stats.values():
                v = jax.lax.pmean(v, "i")
            # instead of computing the mean, we'll just pull everything down and
            #             fiducial_state = jax.lax.pmean(fiducial_state, "i")
            #             derivatives_state = jax.lax.pmean(derivatives_state, "i")
            return (L, stats, fiducial_state, derivatives_state), self.optimizer.update(
                gradient, self.optimizer_state
            )

        (L, stats, fiducial_state, derivatives_state), (
            updates,
            new_state,
        ) = train_step(fiducial, derivatives, noise_fiducial, noise_derivatives)

        if (i + 1) % self.iterations_per_epoch == 0:
            self.history["loss"].append(np.array(strip_array(L), dtype=np.float32))
            stats = strip_array(stats)
            for k, v in stats.items():
                self.history["stats"][k].append(np.array(v, dtype=np.float32))

        if not jnp.sum(jnp.isnan(L)):
            self.optimizer_state = strip_array(new_state)
            self.model_params = optax.apply_updates(
                self.model_params, strip_array(updates)
            )
            #             print("FIDUCIAL STATE\n", tree_map(lambda x: x.shape, fiducial_state))
            #             print("DERIVATIVES STATE\n", tree_map(lambda x: x.shape, derivatives_state))

            #             # NON-EXACT AVERAGING
            #             # average over gpu-dim(0)
            #             fiducial_state = tree_map(lambda x: jnp.mean(x, axis = (0,)), fiducial_state)
            #             # average over gpu-dim(0), +- dim(2), params-dim(3)
            #             # axis 1 is in between because of vmap stacking `out_axes = 1`
            #             derivatives_state = tree_map(lambda x: jnp.mean(x, axis = (0, 2, 3)), derivatives_state)

            # #             print("FIDUCIAL STATE AFTER\n", tree_map(lambda x: x.shape, fiducial_state))
            # #             print("DERIVATIVES STATE AFTER\n", tree_map(lambda x: x.shape, derivatives_state))
            #             # should there be a correction as derivatives_state took much more data into account?
            #             self.model_state = tree_map(lambda x, y: x + y / 2, fiducial_state, derivatives_state)

            # EXACT AVERAGING
            self.model_state = tree_map(
                partial(self.average_xy, x_axis=(0,), y_axis=(0, 2, 3)),
                fiducial_state,
                derivatives_state,
            )

        return i, None

    def train(
        self,
        epochs=10,
        verbose=True,
        verbose_frequency=1,
        save=False,
        save_frequency=5,
        save_filename="imnn_{}",
    ):
        N_epochs_trained = len(self.history["loss_val"])
        for ep in range(N_epochs_trained, N_epochs_trained + epochs):
            if self.prefetch_data is True:
                (
                    fiducial,
                    fiducial_val,
                    noise_fiducial,
                    derivatives,
                    derivatives_val,
                    noise_derivatives,
                ) = (
                    self._finite_it(x)
                    for x in (
                        self.fiducial,
                        self.fiducial_val,
                        self.noise_fiducial,
                        self.derivatives,
                        self.derivatives_val,
                        self.noise_derivatives,
                    )
                )
            else:
                noise_fiducial, noise_derivatives = self._permute(
                    self.noise_fiducial, self.noise_derivatives
                )
                fiducial, derivatives = self._permute(self.fiducial, self.derivatives)
                fiducial_val, derivatives_val = self.fiducial_val, self.derivatives_val

            #             jax.lax.scan(
            #                 lambda i, data: self._update_optimizer(i, *data),
            #                 ep * it_per_epoch,
            #                 (fiducial, noise_fiducial, derivatives, noise_derivatives),
            #                 unroll = 1,
            #             )

            for it, data in enumerate(
                zip(fiducial, noise_fiducial, derivatives, noise_derivatives)
            ):
                self._update_optimizer(ep * self.iterations_per_epoch + it, *data)

            L_val, stats_val = self.loss(
                self.model_params, self.model_state, fiducial_val, derivatives_val
            )
            self.history["loss_val"].append(np.array(L_val, dtype=np.float32))
            for k, v in stats_val.items():
                self.history["stats_val"][k].append(np.array(v, dtype=np.float32))

            if verbose and (ep + 1) % verbose_frequency == 0:
                self.print_current_stats()
            if save and (ep + 1) % save_frequency == 0:
                self.save(
                    filename=save_filename.format(ep + 1),
                    compression={},
                )


# seems to be more or less done, check dimensions for model_state
class IMNN_SGD_collect(IMNN_SGD):
    """Stochastic gradient descent IMNN.
    During training, the loss function and gradients are calculated
    per full batch, which is a collection of batches across all GPUs.
    For validation, a full aggregated Fisher is calculated.
    """

    def __init__(self, *args, **kwargs):
        super(IMNN_SGD_collect, self).__init__(*args, **kwargs)

    #     @partial(jax.jit, backend = "gpu", static_argnums = 0)
    def train_loss(
        self, params, state, fiducial, derivatives, noise_fiducial, noise_derivatives
    ):
        @partial(jax.pmap, devices=jax.devices("gpu"), backend="gpu")
        def collect_predictions_fid(x, noise):
            return self.model.apply(
                {"params": params, **state},
                x + noise,
                **self.model_kwargs,
                use_running_average=False,
            )

        @partial(jax.pmap, devices=jax.devices("gpu"), backend="gpu")
        @partial(jax.vmap, in_axes=(1, None), out_axes=1)
        @partial(jax.vmap, in_axes=(1, None), out_axes=1)
        def collect_predictions_der(x, noise):
            return self.model.apply(
                {"params": params, **state},
                x + noise,
                **self.model_kwargs,
                use_running_average=False,
            )

        summaries_fiducial, fiducial_state = collect_predictions_fid(
            fiducial, noise_fiducial
        )
        #         print("FIDUCIAL STATE\n", tree_map(lambda x: x.shape, fiducial_state))

        summaries_fiducial = jnp.concatenate(summaries_fiducial, axis=0)
        #         fiducial_state = tree_map(lambda x: jnp.mean(jnp.stack(x), axis = 0), fiducial_state)

        summaries_derivatives, derivatives_state = collect_predictions_der(
            derivatives, noise_derivatives
        )
        #         print("DERIVATIVES STATE\n", tree_map(lambda x: x.shape, derivatives_state))

        summaries_derivatives = jnp.concatenate(summaries_derivatives, axis=0)
        #         derivatives_state = tree_map(lambda x: jnp.mean(jnp.stack(x), axis = (0, 2, 3)))

        #         print("FIDUCIAL STATE AFTER\n", tree_map(lambda x: x.shape, fiducial_state))
        #         print("DERIVATIVES STATE AFTER\n", tree_map(lambda x: x.shape, derivatives_state))

        #         new_state = tree_map(lambda x, y: x + y / 2, fiducial_state, derivatives_state)
        #         print("SUMMARIES FIDUCIAL:", summaries_fiducial.shape)
        #         print("SUMMARIES DERIVATIVES:", summaries_derivatives.shape)

        average_state = lambda x, y: self.average_xy(
            jnp.stack(x), jnp.stack(y), (0,), (0, 2, 3)
        )

        # EXACT AVERAGING
        new_state = tree_map(average_state, fiducial_state, derivatives_state)

        δθ = jnp.reshape(self.δθ, (1, self.n_params, 1))
        #         print("delta_theta:", δθ.shape)
        summaries_derivatives = (
            summaries_derivatives[:, 1, ...] - summaries_derivatives[:, 0, ...]
        ) / δθ
        #         print("SUMMARIES DERIVATIVES LATER:", summaries_derivatives.shape)

        F, C, invC, dμ_dθ = self.F_statistics_train(
            summaries_fiducial, summaries_derivatives
        )
        lndetF = self.slogdet(F)
        Λ2 = self.regularization(C, invC)
        r = self.regularization_strength(Λ2)
        L = (
            -lndetF
            + r * Λ2
            + self.l2_norm(params) * self.weight_regularization_strength
        )

        aux = {}
        for k, v in zip(["F", "C", "Cinv", "r", "Λ2"], [F, C, invC, r, Λ2]):
            #             aux[k] = jnp.array(v, dtype = jnp.float32)
            aux[k] = v
        return L, (aux, new_state)

    def _update_optimizer(
        self, i, fiducial, noise_fiducial, derivatives, noise_derivatives
    ):
        #         params = self.optimizer["get_params"](self.optimizer["state"])

        (L, (stats, new_state)), gradient = jax.value_and_grad(
            self.train_loss, has_aux=True
        )(
            self.model_params,
            self.model_state,
            fiducial,
            derivatives,
            noise_fiducial,
            noise_derivatives,
        )

        if (i + 1) % self.iterations_per_epoch == 0:
            self.history["loss"].append(L)
            for k, v in stats.items():
                self.history["stats"][k].append(v)

        if not jnp.sum(jnp.isnan(L)):
            updates, self.optimizer_state = self.optimizer.update(
                gradient, self.optimizer_state
            )
            self.model_params = optax.apply_updates(self.model_params, updates)
            self.model_state = new_state

        return i, None


# left for better times to adapt...
class IMNN_SGD_pmap_params(IMNN_SGD):
    """Stochastic gradient descent IMNN.
    During training, the loss function and gradients are calculated per batch as a collection across all GPUs.
    For validation, a full aggregated Fisher is calculated.
    The difference here is that model parameters are copied across replicas.
    """

    def __init__(self, *args, **kwargs):
        super(IMNN_SGD_pmap_params, self).__init__(*args, **kwargs)

        self.optimizer_state_replicated = replicate_array(
            self.optimizer_state, len(self.gpus)
        )
        self.model_params_replicated = replicate_array(
            self.model_params, len(self.gpus)
        )
        self.model_state_replicated = replicate_array(self.model_state, len(self.gpus))

    def load(self, *args, **kwargs):
        super(IMNN_SGD_pmap_params, self).load(*args, **kwargs)
        self.optimizer_state_replicated = replicate_array(
            self.optimizer_state, len(self.gpus)
        )
        self.model_params_replicated = replicate_array(
            self.model_params_replicated, len(self.gpus)
        )
        self.model_state_replicated = replicate_array(self.model_state, len(self.gpus))

    @partial(jax.jit, backend="gpu", static_argnums=0)
    def train_loss(
        self, params, fiducial, derivatives, noise_fiducial, noise_derivatives
    ):
        def collect_predictions_fid(x, noise):
            return self.model.apply(params, x + noise, **self.model_kwargs)

        @partial(jax.vmap, in_axes=(1, None), out_axes=1)
        @partial(jax.vmap, in_axes=(1, None), out_axes=1)
        def collect_predictions_der(x, noise):
            return self.model.apply(params, x + noise, **self.model_kwargs)

        #         noise_fiducial, noise_derivatives = noise
        #         print(fiducial.shape, noise_fiducial.shape)
        summaries_fiducial = collect_predictions_fid(fiducial, noise_fiducial)
        summaries_derivatives = collect_predictions_der(derivatives, noise_derivatives)
        #         print(summaries_fiducial.shape, summaries_derivatives.shape)

        #         summaries_fiducial = summaries_fiducial.reshape((self.n_sims // 2, self.n_summaries))
        #         summaries_derivatives = summaries_derivatives.reshape((self.n_derivatives // 2, 2, self.n_params, self.n_summaries))
        δθ = jnp.reshape(self.δθ, (1, self.n_params, 1))
        summaries_derivatives = (
            summaries_derivatives[:, 1, ...] - summaries_derivatives[:, 0, ...]
        ) / δθ

        F, C, invC, dμ_dθ = self.F_statistics_train(
            summaries_fiducial, summaries_derivatives
        )
        lndetF = self.slogdet(F)
        Λ2 = self.regularization(C, invC)
        r = self.regularization_strength(Λ2)
        L = (
            -lndetF
            + r * Λ2
            + self.l2_norm(params) * self.weight_regularization_strength
        )

        aux = {}
        for k, v in zip(["F", "C", "Cinv", "r", "Λ2"], [F, C, invC, r, Λ2]):
            #             aux[k] = jnp.array(v, dtype = jnp.float32)
            aux[k] = v
        return L, aux

    def _update_optimizer(
        self, i, fiducial, noise_fiducial, derivatives, noise_derivatives
    ):
        #         params = self.optimizer["get_params"](self.optimizer["state_replicated"])

        @partial(jax.pmap, axis_name="i", devices=jax.devices("gpu"), backend="gpu")
        def train_step(
            fiducial,
            derivatives,
            noise_fiducial,
            noise_derivatives,
            model_params,
            optimizer_state,
        ):
            #             params = self.optimizer["get_params"](state)
            (L, stats), gradient = jax.value_and_grad(self.train_loss, has_aux=True)(
                model_params, fiducial, derivatives, noise_fiducial, noise_derivatives
            )
            gradient = jax.lax.pmean(gradient, "i")
            L = jax.lax.pmean(L, "i")
            for v in stats.values():
                v = jax.lax.pmean(v, "i")
            return (L, stats), self.optimizer.update(gradient, optimizer_state)

        (L, stats), (updates, new_state) = train_step(
            fiducial,
            derivatives,
            noise_fiducial,
            noise_derivatives,
            self.model_params_replicated,
            self.optimizer_state_replicated,
        )

        if (i + 1) % self.iterations_per_epoch == 0:
            self.history["loss"].append(np.array(strip_array(L), dtype=np.float32))
            stats = strip_array(stats)
            for k, v in stats.items():
                self.history["stats"][k].append(np.array(v, dtype=np.float32))

        if not jnp.sum(jnp.isnan(L)):
            self.optimizer_state_replicated = new_state
            self.model_params_replicated = optax.apply_updates(
                self.model_params_replicated, updates
            )

            if (i + 1) % self.iterations_per_epoch == 0:
                self.optimizer_state = strip_array(new_state)
                self.model_params = optax.apply_updates(
                    self.model_params, strip_array(updates)
                )

        return i, None
