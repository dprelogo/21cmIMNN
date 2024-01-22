"""
Base class for Information Maximizing neural networks.
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
import flax
import tensorflow as tf


class IMNNBase:
    """Base class for IMNN.

    Args:
        key: initial `jax.random` key.
        fiducial: fiducial dataset of shape `(n_sims,) + input_dim`.
        derivatives: simulations around fiducial of shape
            `(n_derivatives, 2, n_params) + input_dim`, where parameters of
            simulations are `θ +- δθ/2`.
        noise: telescope noise realizations, of shape `(n_noise,) + input_dim`,
            where `n_noise` should be larger or equal to `max(n_sims, n_derivatives)`.
        θ: fiducial parameter values.
        δθ: total difference in parameters to calculate derivative.
        input_dim: tuple describing input dimensions.
        n_summaries: number of IMNN summaries.
        batch_size_fid: batch size for the fiducial dataset, `n_sims` should be
            divisible by `batch_size_fid * len(devices)`.
        batch_size_der: batch size for the derivatives dataset. `n_derivatives`
            should be divisible by `batch_size_der * len(devices)`. By default,
            it is set so that the number of iterations over `fiducial` and
            `derivatives` are the same.
        n_sims: number of fiducial simulations. Defaults to `fiducial.shape[0]`.
        n_derivatives: number of derivatives. Defaults to `derivatives.shape[0]`.
        n_params: number of fiducial parameters. Defaults to `len(θ)`.
        model_kwargs: neccessary only for `flax.linen` model, where it defines
            mutable parameters, e.g. `{"mutable": ["batch_params"]}`.
        weight_regularization_strength: l2 regularization on all of the model weights.
        C_regularization_strength: controls strength of the covariance
            regularization term. By setting it to 0, regularization is turned off.
        λ: parameter in the covariance regularization.
        ε: parameter in the covariance regularization.
        normalization_factor: if given, data is divided by this factor.
            If `None`, it is calculated from the training set.
        gpus: gpu devices on which the model will run.
            Code assumes only GPU usage. Defaults to `jax.devices("gpu")`
        prefetch_data (bool): if False, data is not prefetched and is only
            manipulated through `jnp.ndarray`.
            If True, data is converted to `tf.data.Dataset` and prefetched
            to the GPUs by `flax.jax_utils.prefetch_to_device`.

    Methods:


    """

    def __init__(
        self,
        key,
        θ,
        δθ,
        input_dim,
        n_summaries,
        model,
        optimizer,
        batch_size_fid,
        fiducial=None,
        derivatives=None,
        noise=None,
        batch_size_der=None,
        n_sims=None,
        n_derivatives=None,
        n_params=None,
        model_kwargs=None,
        weight_regularization_strength=1e-9,
        C_regularization_strength=1.0,
        λ=10.0,
        ε=1.0,
        normalization_factor=None,
        gpus=jax.devices(),
        prefetch_data=True,
    ):
        self.gpus = gpus
        self._key = key
        self.θ, self.δθ = θ, δθ
        self.n_sims = n_sims if n_sims is not None else fiducial.shape[0]
        self.n_derivatives = (
            n_derivatives if n_derivatives is not None else derivatives.shape[0]
        )
        self.n_params = n_params if n_params is not None else len(θ)
        self.n_summaries = n_summaries
        self.input_dim = input_dim
        self.batch_size_fid = batch_size_fid
        if batch_size_der is None:
            self.batch_size_der = (
                self.n_derivatives * self.batch_size_fid // self.n_sims
            )
        else:
            self.batch_size_der = batch_size_der
        self.weight_regularization_strength = weight_regularization_strength
        self.C_regularization_strength = C_regularization_strength
        self.λ = λ
        self.ε = ε

        # data formatting
        self.prefetch_data = prefetch_data
        # if None in [fiducial, derivatives, noise]:
        if any(x is None for x in [fiducial, derivatives, noise]):
            print(
                "As data is not passed, skipping data formatting. "
                "Without it the training will not work!"
            )
        else:
            self._format_data(fiducial, derivatives, noise, normalization_factor)
        self.indexes_fiducial = jnp.arange(self.n_sims // 2)
        self.indexes_derivatives = jnp.arange(self.n_derivatives // 2)

        self.iterations_per_epoch = min(
            self.n_sims // 2 // len(self.gpus) // self.batch_size_fid,
            self.n_derivatives // 2 // len(self.gpus) // self.batch_size_der,
        )

        self._initialize_model(model, model_kwargs)
        self._initialize_optimizer(optimizer)

        self.history = dict(
            loss=[],
            loss_val=[],
            stats=dict(F=[], C=[], Cinv=[], r=[], Λ2=[]),
            stats_val=dict(F=[], C=[], Cinv=[], r=[], Λ2=[]),
        )

    def _get_key(self, num=1):
        """Internal random key generator."""
        self._key, *key = jax.random.split(self._key, num + 1)
        return key if num > 1 else key[0]

    def _initialize_model(self, model, model_kwargs=None):
        """Initializing model parameters and state.

        Args:
            model: `flax.linen` or `jax.example_libraries.stax` model.
            model_kwargs: optional arguments passed to the model while calling
                `model.apply`. Only used by `flax`.

        Returns:
            None
        """
        pass

    def _initialize_optimizer(self, optimizer):
        """Initalizing the optimizer's state.
        Function assumes optimizer from`jax.example_libraries.optimizers`
        with `jax.example_libraries.stax` model and `optax` optimizer for `flax.linen` model.

        Args:
            optimizer: `jax.example_libraries.stax` or `optax` optimizer.

        Returns:
            None
        """
        pass

    def _format_data(self, fiducial, derivatives, noise, normalization_factor):
        """Formatting input data. This includes:
        - splitting the data in half to train/validation,
        - applying noise to the validation set,
        - normalizing,
        - batching / reshaping.

        In case `self.prefetch is True`, all data is properly shuffled, batched
        and prefetched to the GPU for training. Otherwise, training data will
        be shuffled and batched during the training.
        """
        # formatting fiducial datasets
        self.fiducial_val = fiducial[self.n_sims // 2 :]
        self.fiducial_val = self.fiducial_val.reshape(
            (
                self.n_sims // 2 // len(self.gpus) // self.batch_size_fid,
                len(self.gpus),
                self.batch_size_fid,
            )
            + self.input_dim
        )
        self.fiducial = fiducial[: self.n_sims // 2, ..., jnp.newaxis]

        # formatting derivatives datasets
        self.derivatives_val = derivatives[self.n_derivatives // 2 :]
        self.derivatives_val = self.derivatives_val.reshape(
            (
                self.n_derivatives // 2 // len(self.gpus) // self.batch_size_der,
                len(self.gpus),
                self.batch_size_der,
                2,
                self.n_params,
            )
            + self.input_dim
        )
        self.derivatives = derivatives[: self.n_derivatives // 2, ..., jnp.newaxis]

        # formatting fiducial noise
        self.noise_fiducial = noise[: self.n_sims]
        noise_fiducial_val = self.noise_fiducial[self.n_sims // 2 :]
        noise_fiducial_val = noise_fiducial_val.reshape(
            (
                self.n_sims // 2 // len(self.gpus) // self.batch_size_fid,
                len(self.gpus),
                self.batch_size_fid,
            )
            + self.input_dim
        )
        self.noise_fiducial = self.noise_fiducial[: self.n_sims // 2, ..., jnp.newaxis]
        self.fiducial_val = self.fiducial_val + noise_fiducial_val
        del noise_fiducial_val

        # formatting derivatives noise
        self.noise_derivatives = noise[: self.n_derivatives]
        noise_derivatives_val = self.noise_derivatives[self.n_derivatives // 2 :]
        noise_derivatives_val = jnp.broadcast_to(
            noise_derivatives_val.reshape(
                (self.n_derivatives // 2, 1, 1) + self.input_dim
            ),
            (self.n_derivatives // 2, 2, self.n_params) + self.input_dim,
        )
        noise_derivatives_val = noise_derivatives_val.reshape(
            (
                self.n_derivatives // 2 // len(self.gpus) // self.batch_size_der,
                len(self.gpus),
                self.batch_size_der,
                2,
                self.n_params,
            )
            + self.input_dim
        )
        self.noise_derivatives = self.noise_derivatives[
            : self.n_derivatives // 2, ..., jnp.newaxis
        ]
        self.derivatives_val = self.derivatives_val + noise_derivatives_val
        del noise_derivatives_val

        # normalize
        self._normalization(normalization_factor)

        # prefetch
        if self.prefetch_data is True:
            self._prefetching()

    def _normalization(self, normalization_factor=None):
        """Normalizing dataset. Data is assumed to be zero-mean, thus is
        only scaled by the variance of the training set.
        """
        if normalization_factor is None:
            # because the full derivatives object is too big for 32-bit operation,
            # here we approximate the calculation
            variance_simulation = jnp.mean(
                jnp.var(
                    jnp.concatenate(
                        [self.derivatives.flatten(), self.fiducial.flatten()]
                    ).reshape(8, -1),
                    axis=-1,
                )
            )
            variance_noise = jnp.var(self.noise_fiducial)
            normalization_factor = jnp.sqrt(
                variance_simulation + variance_noise
            ).astype(jnp.float32)
        elif normalization_factor < 0:
            raise ValueError(
                f"Normalization factor should be positive, \
                    but its value is {normalization_factor}."
            )
        elif abs(normalization_factor - 1.0) < 1e-4:
            return None

        else:
            normalization_factor = jnp.float32(normalization_factor)

        self.fiducial = self.fiducial / normalization_factor
        self.noise_fiducial = self.noise_fiducial / normalization_factor
        self.fiducial_val = self.fiducial_val / normalization_factor

        self.derivatives = self.derivatives / normalization_factor
        self.noise_derivatives = self.noise_derivatives / normalization_factor
        self.derivatives_val = self.derivatives_val / normalization_factor

    def _prefetching(self, n_prefetch=2):
        """Constructing infinite iterator of the data prefetched to GPUs.
        We use `tensorflow.dataset` to construct infinite shuffled  and batched data,
        followed by `flax.jax_util.prefetch_to_device` to prefetch the data to device.
        """

        def batch_shuffle_prefetch(d, batch_size=None, shuffle=True):
            d = tf.data.Dataset.from_tensor_slices(d)
            if shuffle is True:
                N_shuffle = len(d)
                d = d.shuffle(N_shuffle)
            if batch_size is not None:
                d = d.batch(batch_size).batch(len(self.gpus))
            d = d.repeat().prefetch(tf.data.AUTOTUNE)
            d = flax.jax_utils.prefetch_to_device(
                d.as_numpy_iterator(), n_prefetch, devices=self.gpus
            )
            return d

        self.fiducial = batch_shuffle_prefetch(self.fiducial, self.batch_size_fid, True)
        self.noise_fiducial = batch_shuffle_prefetch(
            self.noise_fiducial, self.batch_size_fid, True
        )
        self.fiducial_val = batch_shuffle_prefetch(self.fiducial_val)

        self.derivatives = batch_shuffle_prefetch(
            self.derivatives, self.batch_size_der, True
        )
        self.noise_derivatives = batch_shuffle_prefetch(
            self.noise_derivatives, self.batch_size_der, True
        )
        self.derivatives_val = batch_shuffle_prefetch(self.derivatives_val)

    def _finite_it(self, iterator):
        """Constructing finite iterator from the infinite one.
        The idea is to iterate over training set / validation set only once.
        With this trick, we can keep continuous caching even between epochs
        on the original (infinite) iterator, with benefit of
        still being able to easily iterate for one epoch.
        """
        for t in range(self.iterations_per_epoch):
            yield next(iterator)

    def save(self, filename="imnn", compression=None):
        """Saving model, optimizer and history.

        Args:
            filename: where to save model/optimizer/history parts.
            compression: `h5py` compression arguments, used for history. For example,
                `compression = {
                    "compression": "gzip",
                    "compression_opts": 7,
                    "shuffle": True,
                    }`
        """
        pass

    def load(self, filename="imnn", load_optimizer=True):
        """Loading the model. Intended usage is to firstly initialize a new
        class instance, followed by loading old model, optimizer and history.
        For example, `imnn = IMNN(*params); imnn.load("imnn")`

        Args:
            filename: from where to load model/optimizer/history.
            load_optimizer (bool): if False, optimizer used for initialization
                is kept. Otherwise, it is overwritten by previously saved one.
        """
        pass

    @staticmethod
    @jax.jit
    def slogdet(matrix):
        lndet = jnp.linalg.slogdet(matrix)
        return lndet[0] * lndet[1]

    @staticmethod
    @jax.jit
    def F_statistics_full(summaries, derivatives):
        C = jnp.cov(summaries, rowvar=False)
        invC = jnp.linalg.inv(C)
        dμ_dθ = jnp.mean(derivatives, axis=0)
        F = jnp.einsum("ij,jk,lk->il", dμ_dθ, invC, dμ_dθ)
        return F, C, invC, dμ_dθ

    @staticmethod
    @jax.jit
    def F_statistics_train(summaries, derivatives):
        C = jnp.cov(summaries, rowvar=False)
        invC = jnp.linalg.inv(C)
        dμ_dθ = jnp.mean(derivatives, axis=0)
        F = jnp.einsum("ik,lk->il", dμ_dθ, dμ_dθ)
        return F, C, invC, dμ_dθ

    @partial(jax.jit, static_argnums=0)
    def regularization_strength(self, Λ2):
        return self.λ * jnp.tanh(Λ2 / self.ε)

    @partial(jax.jit, static_argnums=0)
    def _old_regularization(self, C, invC):
        return jnp.linalg.norm(C - jnp.eye(self.n_summaries)) + jnp.linalg.norm(
            invC - jnp.eye(self.n_summaries)
        )

    @partial(jax.jit, static_argnums=0)
    def regularization(self, C, invC):
        return jnp.linalg.norm(C - jnp.eye(self.n_summaries))

    @staticmethod
    @jax.jit
    def l2_norm(pytree):
        leaves, _ = tree_flatten(pytree)
        return sum(jnp.vdot(x, x) for x in leaves)

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def average_xy(x, y, x_axis, y_axis):
        """Computing weighted average of two arrays, where average has to be
        computed over arbitrary axes, and weights are proportional to the
        number of items.
        For example, if `x.shape == (2, 7, 5, 7)` and `y.shape == (3, 7, 7)`,
        where the object of interest has shape `(7, 7)`, function would return
        `avg = jnp.mean(x, axis = (0, 2)) * 10 / 13 + jnp.mean(y, axis = 0) * 3 / 13`.

        Args:
            x, y: two arrays
            x_axis, y_axis: axes of two arrays over which average is computed.

        Returns:
            weighted average of two arrays.

        """
        x_ave, y_ave = jnp.mean(x, axis=x_axis), jnp.mean(y, axis=y_axis)
        x_s, y_s = x.shape, y.shape
        x_multip = jnp.prod(jnp.array([x_s[i] for i in x_axis]))
        y_multip = jnp.prod(jnp.array([y_s[i] for i in y_axis]))
        w_x, w_y = x_multip / (x_multip + y_multip), y_multip / (x_multip + y_multip)
        return x_ave * w_x + y_ave * w_y

    def print_current_stats(self):
        """Print last stats saved in history."""
        print(
            "Loss: {}, {} | F: {}, {} | C: {}, {} | Cinv: {}, {} | r: {}, {} | Λ2: {}, {}".format(
                self.history["loss"][-1],
                self.history["loss_val"][-1],
                self.slogdet(self.history["stats"]["F"][-1]),
                self.slogdet(self.history["stats_val"]["F"][-1]),
                jnp.linalg.norm(
                    self.history["stats"]["C"][-1] - jnp.eye(self.n_summaries)
                ),
                jnp.linalg.norm(
                    self.history["stats_val"]["C"][-1] - jnp.eye(self.n_summaries)
                ),
                jnp.linalg.norm(
                    jnp.linalg.inv(self.history["stats"]["C"][-1])
                    - jnp.eye(self.n_summaries)
                ),
                jnp.linalg.norm(
                    jnp.linalg.inv(self.history["stats_val"]["C"][-1])
                    - jnp.eye(self.n_summaries)
                ),
                self.history["stats"]["r"][-1],
                self.history["stats_val"]["r"][-1],
                self.history["stats"]["Λ2"][-1],
                self.history["stats_val"]["Λ2"][-1],
            )
        )

    def _permute(self, fid, der):
        """Manual permutation of the arrays at the beginning of each epoch.
        Used only if `prefetch == False`.
        """
        fiducial_shape = (
            self.n_sims // 2 // len(self.gpus) // self.batch_size_fid,
            len(self.gpus),
            self.batch_size_fid,
        )
        derivatives_shape = (
            self.n_derivatives // 2 // len(self.gpus) // self.batch_size_der,
            len(self.gpus),
            self.batch_size_der,
        )

        indexes_fiducial = jax.random.permutation(
            self._get_key(), self.indexes_fiducial
        ).reshape(fiducial_shape)
        indexes_derivatives = jax.random.permutation(
            self._get_key(), self.indexes_derivatives
        ).reshape(derivatives_shape)

        return fid[indexes_fiducial], der[indexes_derivatives]

    def loss(self, *args, **kwargs):
        """IMNN loss function, computed on the whole dataset (one epoch)."""
        pass

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
        """One update iteration - either over whole dataset or one batch,
        depending on the training scheduler.
        """
        pass

    def train(
        self,
        epochs=10,
        verbose=True,
        verbose_frequency=1,
        clear_cache=True,
        clear_cache_frequency=100,
        save=False,
        save_frequency=5,
        save_filename="imnn_{}",
    ):
        """Run the model training.

        Args:
            epochs: number of epochs.
            verbose: to print stats or not.
            verbose_frequency: how often (in #epochs) to print stats.
            clear_cache: to clear xla cache or not. It should help with
                a small memory leak due to `jax.jit`.
            clear_cache_frequency: how often (in #epochs) to clear the cache.
            save: to save or not to save.
            save_frequency: how often (in #epochs) to save model/opt/history.
            save_filename: where to save the model. It is supposed to
                recieve one parameter - epoch, i.e. `save_filename.format(epoch)`.
        """
        pass

    def predict(
        self,
        data,
        batch_size=1,
    ):
        """Predicting the data with the model

        Args:
            data: data to predict.
            batch_size: batch size for the prediction.
        """
        pass
