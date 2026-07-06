import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from dataclasses import dataclass



@dataclass
class StimulusEncodingPlotter:
    CD_dotproduct_avg: np.ndarray
    CD_sample_1_distance_avg: np.ndarray
    CD_sample_2_distance_avg: np.ndarray
    CD_sample_1_selective_avg: np.ndarray   # shape: (n_files, nthresh)
    CD_sample_2_selective_avg: np.ndarray   # shape: (n_files, nthresh)
    thresh: np.ndarray
    ID_P_along_CD1: np.ndarray              # shape: (..., n_files, n_time)
    ID_A_along_CD1: np.ndarray              # shape: (..., n_files, n_time)
    tix_delay: int
    save_dir: str = "figure/stimulus"

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.nthresh = len(self.thresh)

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        return np.corrcoef(x, y)[0, 1]

    @staticmethod
    def _mean_along_first_axis(x: np.ndarray, tix: int) -> np.ndarray:
        """
        Assumes x has shape (something, n_files, n_time),
        matching your use of np.mean(x[:, :, tix], axis=0).
        """
        return np.mean(x[:, :, tix], axis=0)

    def plot_cd_similarity_vs_distance(self, save=True):
        corr1 = self._corr(self.CD_dotproduct_avg, self.CD_sample_1_distance_avg)
        corr2 = self._corr(self.CD_dotproduct_avg, self.CD_sample_2_distance_avg)

        fig = plt.figure(figsize=(4, 4))
        ax1 = plt.subplot(211)
        ax1.scatter(self.CD_dotproduct_avg, self.CD_sample_1_distance_avg)
        ax1.set_title(f"Context 1, cor {corr1:.3f}")
        ax2 = plt.subplot(212)
        ax2.scatter(self.CD_dotproduct_avg, self.CD_sample_2_distance_avg)
        ax2.set_title(f"Context 2, cor {corr2:.3f}")
        ax2.set_xlabel("CD similarity")
        ax2.set_ylabel("Distance btw stim response")
        plt.tight_layout()

        if save:
            fig.savefig(os.path.join(self.save_dir, "CDsimilarity_distance.pdf"))

        return fig

    def plot_cd_similarity_vs_frac_selective(self, context=1, save=True):
        if context == 1:
            selective = self.CD_sample_1_selective_avg
            filename = "CDsimilarity_fracSelective_1.pdf"
        elif context == 2:
            selective = self.CD_sample_2_selective_avg
            filename = "CDsimilarity_fracSelective_2.pdf"
        else:
            raise ValueError("context must be 1 or 2")

        fig = plt.figure(figsize=(6, 6))

        for thx in range(self.nthresh):
            ax = plt.subplot(2, 2, thx + 1)
            corr = self._corr(self.CD_dotproduct_avg, selective[:, thx])
            ax.plot(
                self.CD_dotproduct_avg,
                selective[:, thx],
                marker="o",
                linestyle="None",
            )
            ax.set_title(
                f"thresh {self.thresh[thx]:.1f}" + r"$\sigma$" + f", cor {corr:.3f}"
            )
            ax.set_ylim([0, 0.65])
            if thx == 2:
                ax.set_xlabel("CD similarity")
                ax.set_ylabel("frac stim selective cells")

        plt.tight_layout()

        if save:
            fig.savefig(os.path.join(self.save_dir, filename))

        return fig

    def plot_neural_state_vs_frac_selective(self, stimtype="P", save=True):
        if stimtype == "P":
            neural_state = self._mean_along_first_axis(self.ID_P_along_CD1, self.tix_delay)
            filename = "NeuralState_fracSelective_1P.pdf"
        elif stimtype == "A":
            neural_state = self._mean_along_first_axis(self.ID_A_along_CD1, self.tix_delay)
            filename = "NeuralState_fracSelective_1A.pdf"

        fig = plt.figure(figsize=(6, 6))
        for thx in range(self.nthresh):
            ax = plt.subplot(2, 2, thx + 1)
            cor = self._corr(neural_state, self.CD_sample_1_selective_avg[:, thx])
            ax.plot(
                neural_state,
                self.CD_sample_1_selective_avg[:, thx],
                marker="o",
                linestyle="None",
            )
            ax.set_title(
                f"thresh {self.thresh[thx]:.1f}" + r"$\sigma$" + f", cor {cor:.3f}"
            )
            if thx == 2:
                ax.set_xlabel(r"$\Delta$Neural state along CD1")
                ax.set_ylabel("frac stim selective cells")
        plt.tight_layout()

        if save:
            fig.savefig(os.path.join(self.save_dir, filename))

        return fig

    def plot_neural_state_vs_distance(self, save=True):
        neural_state_P = self._mean_along_first_axis(self.ID_P_along_CD1, self.tix_delay)
        neural_state_A = self._mean_along_first_axis(self.ID_A_along_CD1, self.tix_delay)

        cor_P = self._corr(neural_state_P, self.CD_sample_1_distance_avg)
        cor_A = self._corr(neural_state_A, self.CD_sample_1_distance_avg)

        fig = plt.figure(figsize=(4, 4))
        ax1 = plt.subplot(211)
        ax1.scatter(neural_state_P, self.CD_sample_1_distance_avg)
        ax1.set_title(f"Posterior, cor {cor_P:.3f}")
        ax2 = plt.subplot(212)
        ax2.scatter(neural_state_A, self.CD_sample_1_distance_avg)
        ax2.set_title(f"Anterior, cor {cor_A:.3f}")
        ax2.set_xlabel(r"$\Delta$Neural state along CD1")
        ax2.set_ylabel("Distance btw stim response")
        plt.tight_layout()

        if save:
            fig.savefig(os.path.join(self.save_dir, "NeuralState_distance.pdf"))

        return fig

    def plot_all(self):
        self.plot_cd_similarity_vs_distance()
        self.plot_cd_similarity_vs_frac_selective(context=1)
        self.plot_cd_similarity_vs_frac_selective(context=2)
        self.plot_neural_state_vs_frac_selective(stimtype="P")
        self.plot_neural_state_vs_frac_selective(stimtype="A")
        self.plot_neural_state_vs_distance()    