import os
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class NeuralStateDistancePlotter:
    CD_dotproduct_avg: np.ndarray
    mouse_group_ranks: np.ndarray
    distance_P_all: np.ndarray
    cor_CDdot_distP_firstGroup: np.ndarray
    cor_CDdot_distP_secondGroup: np.ndarray
    ID_P_along_TD1: np.ndarray
    ID_A_along_TD1: np.ndarray
    tvec: np.ndarray
    tsample: float
    tdelay: float
    tresponse: float
    tix_sample: int
    tix_delay: int
    tix_response: int
    firstGroup: np.ndarray
    secondGroup: np.ndarray

    save_dir: str = "figure/neuralstate"

    cmap: any = field(init=False)
    norm: any = field(init=False)
    tix_2: int = field(init=False)
    tix_5: int = field(init=False)

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)

        self.CD_dotproduct_avg = np.asarray(self.CD_dotproduct_avg)
        self.mouse_group_ranks = np.asarray(self.mouse_group_ranks)
        self.distance_P_all = np.asarray(self.distance_P_all)
        self.cor_CDdot_distP_firstGroup = np.asarray(self.cor_CDdot_distP_firstGroup)
        self.cor_CDdot_distP_secondGroup = np.asarray(self.cor_CDdot_distP_secondGroup)
        self.ID_P_along_TD1 = np.asarray(self.ID_P_along_TD1)
        self.ID_A_along_TD1 = np.asarray(self.ID_A_along_TD1)
        self.tvec = np.asarray(self.tvec)
        self.firstGroup = np.asarray(self.firstGroup)
        self.secondGroup = np.asarray(self.secondGroup)

        self.cmap = plt.cm.jet
        self.norm = plt.Normalize(self.mouse_group_ranks.min(), self.mouse_group_ranks.max())
        self.tix_2 = np.where(self.tvec >= 2)[0][0]
        self.tix_5 = np.where(self.tvec >= 5)[0][0]

    def _save(self, fig, name, save):
        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(self.save_dir, name))

    def plot_distance_scatter_time(self, save):
        time_points = [0, self.tix_sample, self.tix_2, self.tix_delay, self.tix_response, self.tix_5]
        titles = ["pre-sample", "sample", "2", "delay", "response", "post-response"]

        fig = plt.figure(figsize=(12, 3))
        for i, tx in enumerate(time_points):
            ax = plt.subplot(1, 6, i + 1)
            y = np.mean(self.distance_P_all[:, :, tx], axis=0)
            colors = self.cmap(self.norm(self.mouse_group_ranks))

            ax.scatter(self.CD_dotproduct_avg, y, marker="o", c=colors)
            ax.axvline(0, color="gray", linestyle="--")
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_xlabel("CD similarity")
            ax.set_title(f"t={titles[i]}")
            if i == 0:
                ax.set_ylabel("Eucl Distance btw neural states")

        self._save(fig, "euclidean_posterior.pdf", save=True)
        return fig

    def plot_distance_corr_time(self, save):
        fig = plt.figure(figsize=(4, 4))

        ax1 = plt.subplot(211)
        ax1.plot(self.tvec, np.mean(self.cor_CDdot_distP_firstGroup, axis=0), c="purple")
        ax1.axvline(self.tsample, color="gray", linestyle="--")
        ax1.axvline(self.tdelay, color="gray", linestyle="--")
        ax1.axvline(self.tresponse, color="gray", linestyle="--")
        ax1.axhline(0, color="gray", linestyle="--")
        ax1.set_title("Mouse Group 1")
        ax1.set_ylim([-0.8, 0.8])

        ax2 = plt.subplot(212)
        ax2.plot(self.tvec, np.mean(self.cor_CDdot_distP_secondGroup, axis=0), c="purple")
        ax2.axvline(self.tsample, color="gray", linestyle="--")
        ax2.axvline(self.tdelay, color="gray", linestyle="--")
        ax2.axvline(self.tresponse, color="gray", linestyle="--")
        ax2.axhline(0, color="gray", linestyle="--")
        ax2.set_title("Mouse Group 2")
        ax2.set_ylim([-0.8, 0.8])
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("cor: CD similarity vs Eucl distance")

        self._save(fig, "euclidean_cor_time.pdf", save=True)
        return fig

    def plot_distance_mean_time(self, save):
        fig = plt.figure(figsize=(4, 3))
        plt.plot(
            self.tvec,
            np.mean(self.distance_P_all[:, self.firstGroup, :], axis=(0, 1)),
            label="Mouse Group 1",
            c="purple",
        )
        plt.plot(
            self.tvec,
            np.mean(self.distance_P_all[:, self.secondGroup, :], axis=(0, 1)),
            label="Mouse Group 2",
            c="purple",
            linestyle="--",
        )
        plt.axvline(self.tsample, color="gray", linestyle="--")
        plt.axvline(self.tdelay, color="gray", linestyle="--")
        plt.axvline(self.tresponse, color="gray", linestyle="--")
        plt.ylim([0.2, 1])
        plt.xlabel("time (s)")
        plt.ylabel("Eucl distance btw neural states")
        plt.legend()

        self._save(fig, "euclidean_mean_time.pdf", save=True)
        return fig

    def plot_transient_direction(self, save):
        yP = np.mean(self.ID_P_along_TD1, axis=0)
        yA = np.mean(self.ID_A_along_TD1, axis=0)

        cor_P = np.corrcoef(self.CD_dotproduct_avg, yP)[0, 1]
        cor_A = np.corrcoef(self.CD_dotproduct_avg, yA)[0, 1]

        fig = plt.figure(figsize=(4.5, 3))
        plt.scatter(self.CD_dotproduct_avg, yP, marker="o", c="purple", label="Pos")
        plt.scatter(self.CD_dotproduct_avg, yA, marker="o", c="limegreen", label="Ant")
        plt.axvline(0, color="gray", linestyle="--")
        plt.axhline(0, color="gray", linestyle="--")
        plt.xlabel("CD similarity")
        plt.ylabel(r"$\Delta$Neural state along TD")
        plt.title(f"cor_P={cor_P:.2f}, cor_A={cor_A:.2f}")
        plt.legend(frameon=False, bbox_to_anchor=[1, 0.8])

        self._save(fig, "TD.pdf", save=True)
        return fig

    def plot_distance_all(self):
        self.plot_distance_scatter_time()
        self.plot_distance_corr_time()
        self.plot_distance_mean_time()

    def plot_all(self):
        self.plot_distance_all()
        self.plot_transient_direction()