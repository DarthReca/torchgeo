import os
import warnings
from dataclasses import dataclass
from datetime import date
from itertools import product
from typing import Any, Literal

import numpy as np
import torch
from einops import rearrange

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_url, lazy_import


@dataclass
class WaterData:
    full_input_timeseries: np.ndarray
    input_timeseries: np.ndarray
    output_timeseries: np.ndarray
    input_cloud_mask: np.ndarray
    output_cloud_mask: np.ndarray
    dem: np.ndarray
    climate: np.ndarray


h5py = lazy_import("h5py")
lazy_import("hdf5plugin")


class HydroChronos(NonGeoDataset):
    all_bands = ("B1", "B2", "B3", "B4", "B5", "B7")
    rgb_bands = ("B3", "B2", "B1")
    climate_vars = (
        "aet",
        "def",
        "pdsi",
        "pet",
        "pr",
        "ro",
        "soil",
        "srad",
        "swe",
        "tmmn",
        "tmmx",
        "vap",
        "vpd",
        "vs",
    )
    urls = (
        "https://huggingface.co/datasets/DarthReca/hydro-chronos/resolve/main/climate.h5",
        "https://huggingface.co/datasets/DarthReca/hydro-chronos/resolve/main/landsat/hydrochronos_landsat_main.h5",
        "https://huggingface.co/datasets/DarthReca/hydro-chronos/blob/main/landsat/hydrochronos_landsat_0.h5",
        "https://huggingface.co/datasets/DarthReca/hydro-chronos/blob/main/landsat/hydrochronos_landsat_1.h5",
        "https://huggingface.co/datasets/DarthReca/hydro-chronos/resolve/main/landsat/hydrochronos_landsat_2.h5",
        "https://huggingface.co/datasets/DarthReca/hydro-chronos/blob/main/sentinel/hydrochronos_sentinel_main.h5",
        "https://huggingface.co/datasets/DarthReca/hydro-chronos/blob/main/sentinel/hydrochronos_sentinel_0.h5",
        "https://huggingface.co/datasets/DarthReca/hydro-chronos/blob/main/sentinel/hydrochronos_sentinel_1.h5",
        "https://huggingface.co/datasets/DarthReca/hydro-chronos/blob/main/sentinel/hydrochronos_sentinel_2.h5",
        "https://huggingface.co/datasets/DarthReca/hydro-chronos/blob/main/sentinel/hydrochronos_sentinel_3.h5",
        "https://huggingface.co/datasets/DarthReca/hydro-chronos/blob/main/sentinel/hydrochronos_sentinel_4.h5"
    )
    filenames = (
        "climate.h5",
        "hydrochronos_landsat_main.h5",
        "hydrochronos_landsat_0.h5",
        "hydrochronos_landsat_1.h5",
        "hydrochronos_landsat_2.h5",
        "hydrochronos_sentinel_main.h5",
        "hydrochronos_sentinel_0.h5",
        "hydrochronos_sentinel_1.h5",
        "hydrochronos_sentinel_2.h5",
        "hydrochronos_sentinel_3.h5",
        "hydrochronos_sentinel_4.h5",
    )
    md5s = (
        "a70bb7e4a2788657c2354c4c3d9296fe",
        "15d78fb825f9a81dad600db828d22c08",
        "15d78fb825f9a81dad600db828d22c08",
        "15d78fb825f9a81dad600db828d22c08",
        "15d78fb825f9a81dad600db828d22c08",
        "15d78fb825f9a81dad600db828d22c08",
        "15d78fb825f9a81dad600db828d22c08",
        "15d78fb825f9a81dad600db828d22c08",
        "15d78fb825f9a81dad600db828d22c08",
        "15d78fb825f9a81dad600db828d22c08",
        "15d78fb825f9a81dad600db828d22c08",
    )

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        satellite: Literal["landsat", "sentinel"] = "landsat",
        absolute_values: bool = False,
        climate_seq_len: int = 5,
        climate_bands: tuple[str, ...] = ("tmmx", "pr", "ro", "soil", "aet"),
        download: bool = False,
        checksum: bool = False,
    ):

        self.absolute_values = absolute_values
        self.root = root
        self.split = split
        self.satellite = satellite
        self.download = download
        self.checksum = checksum
        # MOCKUP
        self.filenames = ["climate.h5",
                          f"hydrochronos_{self.satellite}_main.h5"]

        # Load splits
        self.input_output_years = self._load_temporal_split()
        self._verify()
        # Load all files name

        self.files = self._load_samples()
        self.climate_bands = climate_bands
        self.seq_len = climate_seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._load_data(*self.files[index])
        sample |= {
            "name": self.files[index][0].split("/")[-1],
            "in_years": self.files[index][1],
            "out_years": self.files[index][2],
        }
        return sample

    def _load_temporal_split(self):
        if self.split == "train":
            return [
                (np.arange(1990, 1995), np.arange(1995, 2000)),
                (np.arange(1995, 2000), np.arange(2000, 2005)),
            ]
        elif self.split == "val":
            return [(np.arange(2000, 2005), np.arange(2005, 2010))]
        else:
            return [(np.arange(2015, 2020), np.arange(2020, 2025))]

    def _load_samples(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        h5py = lazy_import("h5py")
        lazy_import("hdf5plugin")

        with h5py.File(f"{self.root}/hydrochronos_{self.satellite}_main.h5", "r") as f:
            keys_years = map(
                lambda k: (k[0], *k[1]),
                product(f.keys(), self.input_output_years)
            )
            files = [
                (k, np.array(in_years), np.array(out_years))
                for k, in_years, out_years in keys_years
                if (
                    k in f
                    and len(np.intersect1d(in_years, f[k].attrs["years"]))
                    >= 0.8 * len(in_years)
                )
            ]
        with h5py.File(f"{self.root}/climate.h5") as f:
            keys = set(f.keys())

        return [k for k in files if k[0] in keys]

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        exists = []
        for filename in self.filenames:
            filepath = os.path.join(self.root, filename)
            exists.append(os.path.exists(filepath))

        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        for url, filename, md5 in zip(self.urls, self.filenames, self.md5s):
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                download_url(
                    url,
                    self.root,
                    filename=filename,
                    md5=md5 if self.checksum else None,
                )

    def _load_data(self, key: str, in_years: np.ndarray, out_years: np.ndarray):
        data = self._load_time_series(key, in_years, out_years)
        # climate = self._load_climate_data(key, years, in_years)
        input_image = data.full_input_timeseries.transpose(1, 0, 2, 3)
        diff_mndwi = self._compute_mask(data)[np.newaxis]
        return {
            "image": input_image,
            "mask": diff_mndwi,
            "dem": data.dem
        }

    def _compute_mndwi(self, ds: np.ndarray) -> np.ndarray:
        ds = ds.astype("float32")
        b5 = self.all_bands.index("B5")
        b2 = self.all_bands.index("B2")
        return (ds[b2] - ds[b5]) / (ds[b2] + ds[b5] + 1e-9)

    def _load_time_series(
        self, key: str, in_years: np.ndarray, out_years: np.ndarray
    ) -> WaterData:
        h5py = lazy_import("h5py")
        lazy_import("hdf5plugin")

        with h5py.File(f"{self.root}/hydrochronos_{self.satellite}_main.h5", "r") as f:
            bands = f[key]["bands"][...]
            cloud_mask = f[key]["cloud_mask"][...]
            dem = f[key]["dem"][...].astype("float32")
            years = f[key].attrs["years"]
        all_input_years = [in_y for in_y, _ in self.input_output_years]
        all_input_years = max(
            all_input_years, key=lambda x: np.intersect1d(in_years, x).size
        )
        in_years_mask = (in_years.min() <= years) & (years <= in_years.max())
        out_years_mask = (out_years.min() <= years) & (
            years <= out_years.max())
        # Create a mask for the invalid pixels
        relevant_bands = [self.all_bands.index(
            "B5"), self.all_bands.index("B2")]
        no_data_mask = (bands[relevant_bands] == 0).any(axis=0)
        unclear_mask = cloud_mask | no_data_mask
        input_cloud_mask = unclear_mask[in_years_mask]
        output_cloud_mask = unclear_mask[out_years_mask]
        # Get the input and output images
        in_series = bands[:, in_years_mask]
        out_series = bands[:, out_years_mask]
        # Zero impute the missing input years
        input_image = np.zeros(
            (in_series.shape[0], len(all_input_years), *in_series.shape[2:]),
            dtype=in_series.dtype,
        )
        for i, year in enumerate(all_input_years):
            if year in years:
                input_image[:, i] = bands[:, years == year].squeeze()
        return WaterData(
            input_image,
            in_series,
            out_series,
            input_cloud_mask,
            output_cloud_mask,
            dem,
            None
        )

    def _compute_mask(self, data: WaterData) -> np.ndarray:
        in_mndwi = self._compute_mndwi(data.input_timeseries)
        out_mndwi = self._compute_mndwi(data.output_timeseries)
        # Remove clouds and shadows
        in_mndwi[data.input_cloud_mask] = np.nan
        out_mndwi[data.output_cloud_mask] = np.nan
        # Compute the difference in MNDWI
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        in_median = np.nanmedian(in_mndwi, axis=0)
        out_median = np.nanmedian(out_mndwi, axis=0)
        diff_mndwi = in_median - out_median
        warnings.filterwarnings("default", message="All-NaN slice encountered")
        diff_mndwi = np.nan_to_num(diff_mndwi, nan=0)
        return diff_mndwi

    def _load_climate_data(self, name: str, end_dates: list[date], in_years: list[int]) -> np.ndarray:
        with h5py.File(f"{self.root}/climate.h5") as f:
            climate_dates = [date.fromisoformat(
                str(t)) for t in f[name]["time"][...]]
            end_indices = [climate_dates.index(d) for d in end_dates]
            start_indices = [max(i - self.seq_len, 0) for i in end_indices]
            climate_indices = np.zeros(f[name]["climate"].shape[1], dtype=bool)
            for start, end in zip(start_indices, end_indices):
                climate_indices[start:end] = True
            climate_bands = f[name]["climate"][:,
                                               climate_indices].astype("float32")
            if climate_bands.shape[1] != len(in_years) * self.seq_len:
                climate_bands = np.concatenate(
                    [climate_bands, np.ones(
                        (climate_bands.shape[0], 1)) * np.nan],
                    axis=1,
                )
        # Select bands
        indexes = [self.climate_vars.index(k) for k in self.climate_bands]
        climate_bands = climate_bands[indexes]
        climate_bands = np.concatenate(
            [climate_bands, np.isnan(climate_bands).any(axis=0)[np.newaxis]]
        )
        np.nan_to_num(climate_bands, copy=False)
        return rearrange(climate_bands, "c (b t) -> b t c", b=self.seq_len)
