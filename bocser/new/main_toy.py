import numpy as np
import gpflow
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from plotnine import *

import trieste
from trieste.acquisition.function import ExpectedImprovement

import timeit
from icecream import install

install()

from imp_var import ImprovementVariance


class ArtSurface:

    def __init__(self,
                 num_minima,
                 activation_barrier=2.28,
                 elevation=5.0,
                 ring_width=0.05):
        self.N = num_minima
        self.activation_barrier = activation_barrier
        self.elevation = elevation
        self.ring_width = ring_width

        # def _f(z):
        #     x, y = np.real(z), np.imag(z)
        #     return (x**2 + y**2)**2 + 4*x*(x**2 + y**2) - 4*y**2
        # def _f(z):
        #     x, y = np.real(z), np.imag(z)
        #     return 2 * (x**4 - x**2 + y**2) + np.exp(-1 / (x**2 + y**2))

        # self.bump_shape = _f
        # self.bump_shape = lambda z: np.abs(np.real(z) + np.imag(z)*1j*2)**2 - 1.0
        self.bump_shape = lambda z: np.abs(z)**2 - 1.0

    def _bump(self, z):
        """This is a bump between [-1.0, 0.0] in shape of self.bump_shape"""
        s = self.bump_shape(z)
        eps = 1e-12
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.exp(-self.ring_width / (s**2 + eps))
            # result[np.isclose(s, 0.0, atol=1e-6)] = 0.0
        return 1.0 - result

    def _angular(self, z):
        """Oscillations at origin between [-0.5, 0.5]"""
        return np.real(z**self.N) / np.abs(z**self.N) / 2 + 0.5

    def __call__(self, x, y):
        z = x + 1j * y
        bump = self._bump(z)
        angular = self._angular(z)

        result = angular * bump * self.activation_barrier + (
            1.0 - bump) * self.elevation
        return result


potential = ArtSurface(10)


class ProgressLogger:

    def __init__(self,
                 potential,
                 xlim=(-2, 2),
                 ylim=(-2, 2),
                 grid_size=500,
                 output_prefix="plot"):
        self.output_prefix = output_prefix
        self.guesses = []

        x = np.linspace(xlim[0], xlim[1], grid_size)
        y = np.linspace(ylim[0], ylim[1], grid_size)
        self.X, self.Y = np.meshgrid(x, y)

        self.V_val = potential(self.X, self.Y)
        self.df_potential = pd.DataFrame({
            'x': self.X.ravel(),
            'y': self.Y.ravel(),
            'V': self.V_val.ravel()
        })

    def log(self, point, step):
        ic(point.numpy())
        self.guesses.append(point.numpy())
        if step == 99:
            guesses_np = np.array(self.guesses)

            df_guesses = pd.DataFrame({
                'x': guesses_np[:, 0],
                'y': guesses_np[:, 1]
            })

            normal_theme = (theme_bw() + theme(
                panel_grid_major=element_blank(),
                panel_grid_minor=element_blank(),
                panel_border=element_rect(colour='black', fill=None, size=1),
                axis_line=element_line(colour='black'),
                axis_title=element_text(size=16, face='bold', ma='center'),
                axis_text=element_text(size=14),
                legend_title=element_text(size=14, face='bold'),
                legend_text=element_text(size=14),
                figure_size=(7 * 0.75, 6 * 0.75)))

            p = (ggplot() +
                 geom_tile(self.df_potential, aes(x='x', y='y', fill='V')) +
                 geom_point(data=df_guesses,
                            mapping=aes(x='x', y='y'),
                            color='red',
                            size=1.5) + scale_fill_gradientn(colors=[
                                '#543005', '#8c510a', '#bf812d', '#dfc27d',
                                '#f6e8c3', '#f5f5f5', '#c7eae5', '#80cdc1',
                                '#35978f', '#01665e', '#003c30'
                            ]) + normal_theme)  # +
            # coord_fixed() + labs(title=f'Progress at step {step}',
            #                      x='Re(z)',
            #                      y='Im(z)',
            #                      fill='V(z)') + theme(figure_size=(6, 5)))

            p.save(f"images/{self.output_prefix}_{step:03d}.png",
                   verbose=False)
            p.save(f"images/{self.output_prefix}_{step:03d}.svg",
                   verbose=False)

    # def log(self, point, step):
    #     ic(point.numpy())
    #     self.guesses.append(point.numpy())

    #     fig, ax = plt.subplots(figsize=(6, 5))
    #     contour = ax.contourf(self.X,
    #                           self.Y,
    #                           self.V_val,
    #                           levels=100,
    #                           cmap='BrBG')
    #     fig.colorbar(contour, ax=ax, label='V(z)')

    #     ax.set_title(f'Progress at step {step}')
    #     ax.set_xlabel('Re(z)')
    #     ax.set_ylabel('Im(z)')
    #     ax.set_aspect('equal')

    #     guesses_np = np.array(self.guesses)
    #     # ic(guesses_np)
    #     ax.scatter(guesses_np[:, 0], guesses_np[:, 1], color='red', s=10)

    #     fig.savefig(f"images/{self.output_prefix}_{step:03d}.png")
    #     plt.close(fig)


def observer(x: tf.Tensor) -> tf.Tensor:
    x_np = x.numpy()
    obs = np.array([[potential(*pt)] for pt in x_np])
    return trieste.data.Dataset(
        query_points=x,
        observations=tf.constant(obs, dtype=tf.float64),
    )


if __name__ == "__main__":
    num_dims = 2
    search_space = trieste.space.Box([-2.0 for _ in range(num_dims)],
                                     [2.0 for _ in range(num_dims)])

    num_initial_points = 5
    initial_query_points = search_space.sample(num_initial_points)
    initial_data = observer(initial_query_points)

    # # Initialize with a few point on a circle
    # initial_data += observer(
    #     tf.constant([
    #         ic([np.real(np.exp(1j * phi)),
    #             np.imag(np.exp(1j * phi))])
    #         for phi in np.linspace(-np.pi, np.pi, 2, endpoint=False)
    #     ],
    #                 dtype=tf.float64))

    gpflow_model = gpflow.models.GPR(
        initial_data.astuple(),
        kernel=gpflow.kernels.RBF(
            variance=0.07,
            lengthscales=0.1,
            active_dims=[i for i in range(num_dims)],
        ),
        noise_variance=1e-4,
    )
    gpflow.set_trainable(gpflow_model.likelihood, False)
    gpflow.set_trainable(gpflow_model.kernel.lengthscales, True)

    model = trieste.models.gpflow.GaussianProcessRegression(gpflow_model)

    ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
        search_space=search_space,
        datasets=initial_data,
        models=model,
        acquisition_rule=trieste.acquisition.rule.EfficientGlobalOptimization(
            # ExpectedImprovement(),
            ImprovementVariance(2.0, circle_informed=True), ))
    ask_tell.acquisition_state
    logger = ProgressLogger(potential,
                            xlim=[-2, 2],
                            ylim=[-2, 2],
                            output_prefix="informed")

    for step in range(100):
        start = timeit.default_timer()
        new_point = ask_tell.ask()
        stop = timeit.default_timer()

        print(
            f"Time at step {step + 1}: {stop - start}; "
            f"Deepest minimum: {tf.reduce_min(ask_tell.dataset.observations).numpy()}"
        )

        new_data = observer(new_point)
        ask_tell.tell(new_data)
        logger.log(new_point[0], step)
