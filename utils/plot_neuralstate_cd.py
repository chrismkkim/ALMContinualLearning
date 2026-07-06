import os
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class NeuralStateAlongCDPlotter:
    CD_dotproduct_avg: np.ndarray
    mouse_group_ranks: np.ndarray
    ID_P_along_CD1: np.ndarray
    ID_A_along_CD1: np.ndarray
    ID_P_along_CD2: np.ndarray
    ID_A_along_CD2: np.ndarray
    cor_CDdot_IDP_CD1: np.ndarray
    cor_CDdot_IDA_CD1: np.ndarray
    cor_CDdot_IDP_CD2: np.ndarray
    cor_CDdot_IDA_CD2: np.ndarray
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

    cfg: dict = field(init=False)

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)

        self.CD_dotproduct_avg = np.asarray(self.CD_dotproduct_avg)
        self.mouse_group_ranks = np.asarray(self.mouse_group_ranks)
        self.tvec = np.asarray(self.tvec)
        self.firstGroup = np.asarray(self.firstGroup)
        self.secondGroup = np.asarray(self.secondGroup)

        self.cmap = plt.cm.jet
        self.norm = plt.Normalize(self.mouse_group_ranks.min(), self.mouse_group_ranks.max())

        self.cfg = {
            (1, "P"): dict(delaydir=self.ID_P_along_CD1, corr=self.cor_CDdot_IDP_CD1, ylim=[-0.7, 0.3], name="posterior"),
            (1, "A"): dict(delaydir=self.ID_A_along_CD1, corr=self.cor_CDdot_IDA_CD1, ylim=[-0.3, 0.7], name="anterior"),
            (2, "P"): dict(delaydir=self.ID_P_along_CD2, corr=self.cor_CDdot_IDP_CD2, ylim=[-0.3, 0.7], name="posterior"),
            (2, "A"): dict(delaydir=self.ID_A_along_CD2, corr=self.cor_CDdot_IDA_CD2, ylim=[-0.7, 0.3], name="anterior"),
        }

    def _save(self, fig, name, save=True):
        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(self.save_dir, name))

    def plot_cd_dotproduct_vs_neuralstate_along_CD(self, cd: int, stimtype: str, save=True):
        c = self.cfg[(cd, stimtype)]
        delaydir = c["delaydir"]
        times = [self.tix_sample, self.tix_delay, self.tix_response]
        titles = ["sample", "delay", "response"]

        fig = plt.figure(figsize=(8, 3))
        for i, tx in enumerate(times):
            ax = plt.subplot(1, 3, i + 1)
            color = self.cmap(self.norm(self.mouse_group_ranks))
            ax.scatter(self.CD_dotproduct_avg, np.mean(delaydir[:, :, tx], axis=0), marker="o", c=color)
            ax.axvline(0, color="gray", linestyle="--")
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_xlabel("CD similarity")
            ax.set_title(f"t={titles[i]}")
            ax.set_ylim(c["ylim"])
            if i == 0:
                ax.set_ylabel(rf"$\Delta$Neural state along CD{cd}")

        self._save(fig, f"CD{cd}_{c['name']}.pdf", save=True)
        return fig

    def plot_corr_time(self, cd: int, save=True):
        fig = plt.figure(figsize=(4, 3))
        plt.plot(self.tvec, np.mean(self.cfg[(cd, "P")]["corr"], axis=0), label="Pos", c="purple")
        plt.plot(self.tvec, np.mean(self.cfg[(cd, "A")]["corr"], axis=0), label="Ant", c="limegreen")
        plt.axvline(self.tsample, color="gray", linestyle="--")
        plt.axvline(self.tdelay, color="gray", linestyle="--")
        plt.axvline(self.tresponse, color="gray", linestyle="--")
        plt.xlabel("time (s)")
        plt.ylabel(r"corr: CD similarity vs $\Delta$Neural state")
        plt.legend()
        self._save(fig, f"CD{cd}_corr_time.pdf", save=True)
        return fig

    def plot_mean_time(self, cd: int, save=True):
        pos = self.cfg[(cd, "P")]["delaydir"]
        ant = self.cfg[(cd, "A")]["delaydir"]

        fig = plt.figure(figsize=(5.5, 3))
        plt.plot(self.tvec, np.mean(pos[:, self.firstGroup, :], axis=(0, 1)), label="Pos Group1", c="purple")
        plt.plot(self.tvec, np.mean(pos[:, self.secondGroup, :], axis=(0, 1)), label="Pos Group2", c="purple", linestyle="--")
        plt.plot(self.tvec, np.mean(ant[:, self.firstGroup, :], axis=(0, 1)), label="Ant Group1", c="limegreen")
        plt.plot(self.tvec, np.mean(ant[:, self.secondGroup, :], axis=(0, 1)), label="Ant Group2", c="limegreen", linestyle="--")
        plt.axvline(self.tsample, color="gray", linestyle="--")
        plt.axvline(self.tdelay, color="gray", linestyle="--")
        plt.axvline(self.tresponse, color="gray", linestyle="--")
        plt.xlabel("time (s)")
        plt.ylabel(rf"$\Delta$Neural state along CD{cd}")
        plt.legend(frameon=False, bbox_to_anchor=[1, 0.8])
        self._save(fig, f"CD{cd}_mean_time.pdf", save=True)
        return fig

    def plot_all(self):
        for cd in [1, 2]:
            for stimtype in ["P", "A"]:
                self.plot_cd_dotproduct_vs_neuralstate_along_CD(cd, stimtype, save=False)
            self.plot_corr_time(cd, save=False)
            self.plot_mean_time(cd, save=False)