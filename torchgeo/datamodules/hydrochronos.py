from typing import Any, Dict

import kornia.augmentation as K
import torch

from .geo import NonGeoDataModule


class D4(K.AugmentationBase2D):

    def apply_transform(self, input: torch.Tensor, params: dict[str, torch.Tensor], flags: dict[str, Any], transform: torch.Tensor | None = None) -> torch.Tensor:
        choice = torch.randint(0, 8, (1,)).item()
        if choice == 0:
            return input                                    # identity
        elif choice == 1:
            return torch.rot90(input, 1, [-2, -1])       # r90
        elif choice == 2:
            return torch.rot90(input, 2, [-2, -1])       # r180
        elif choice == 3:
            return torch.rot90(input, 3, [-2, -1])       # r270
        elif choice == 4:
            return input.flip(-1)                        # h-flip
        elif choice == 5:
            return input.flip(-2)                        # v-flip
        elif choice == 6:
            return torch.rot90(input.flip(-1), 1, [-2, -1])   # transpose
        else:
            return torch.rot90(input.flip(-1), 3, [-2, -1])   # transverse


class HydroChronosDataModule(NonGeoDataModule):
    mean = {
        'landsat': torch.tensor([31.510, 29.306, 25.032, 62.795, 37.586, 21.682, 4.845]),
        'sentinel': torch.tensor([27.617, 24.782, 19.763, 60.604, 38.653, 21.459, 4.845]),
    }
    std = {
        'landsat': torch.tensor([18.089, 27.216, 27.928, 34.766, 26.263, 21.155, 0.848]),
        'sentinel': torch.tensor([20.668, 20.661, 23.357, 33.452, 25.807, 19.753, 0.848]),
    }

    def __init__(self, cls, num_workers: int = 0, batch_size: int = 1, **kwargs):
        super().__init__(cls, num_workers=num_workers, batch_size=batch_size, **kwargs)
        self.train_aug = K.AugmentationSequential(
            D4(),
            K.RandomAffine(degrees=0,                            translate=(
                0.1, 0.1), scale=(0.5, 1.0), p=1.0),
            K.Resize(size=(256, 256)),
            data_keys=None
        )


def standardize_climate_var(climate_bands: np.ndarray) -> np.ndarray:
    # Fill missing values
    climate_bands[climate_bands == -32768] = np.nan
    # Precipitation bands (log1p + Standard scaling)
    index = CLIMATE_VARS.index("pr")
    climate_bands[index] = (np.log1p(climate_bands[index]) - 4.16) / 1.05
    # Max temp band (Standard scaling)
    index = CLIMATE_VARS.index("tmmx")
    climate_bands[index] = (climate_bands[index] - 192.2) / 120.0
    # Min temp band (Standard scaling)
    index = CLIMATE_VARS.index("tmmn")
    climate_bands[index] = (climate_bands[index] - 86.2) / 111.15
    # Actual Evapotranspiration band (log1p + Standard scaling)
    index = CLIMATE_VARS.index("aet")
    climate_bands[index] = (np.log1p(climate_bands[index]) - 5.45) / 2.25
    # Runoff band (log1p + Standard scaling)
    index = CLIMATE_VARS.index("ro")
    climate_bands[index] = (np.log1p(climate_bands[index]) - 2.19) / 1.68
    # Soil moisture band (log1p + Standard scaling)
    index = CLIMATE_VARS.index("soil")
    climate_bands[index] = (np.log1p(climate_bands[index]) - 6.42) / 1.09
    return climate_bands
