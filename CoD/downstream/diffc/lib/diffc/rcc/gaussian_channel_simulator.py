import torch
from downstream.diffc.lib.diffc.rcc.chunk_coding import (
    get_chunk_sizes,
    chunk_and_encode,
    decode_from_chunks,
)
import numpy as np
from zipf_encoding import encode_zipf, decode_zipf


class GaussianChannelSimulator:
    def __init__(self, max_chunk_size, chunk_padding):
        self.max_chunk_size = max_chunk_size
        self.chunk_padding = chunk_padding

    def encode(self, mu, manual_dkl=None, seed=0):
        """Simulates a noisy channel with identity covariance and mean mu."""
        dkl = manual_dkl
        if dkl is None:
            dkl = 0.5 * float((mu.astype(np.float32) ** 2).sum() / np.log(2))
        print(manual_dkl, dkl)

        chunk_sizes = get_chunk_sizes(dkl, self.max_chunk_size, self.chunk_padding)
        chunk_seeds, sample = chunk_and_encode(
            mu, chunk_sizes=chunk_sizes, shared_seed=seed
        )

        return sample, chunk_seeds, dkl

    def decode(self, chunk_seeds, dim, dkl, seed=0):
        chunk_sizes = get_chunk_sizes(dkl, self.max_chunk_size, self.chunk_padding)
        return decode_from_chunks(dim, chunk_seeds, chunk_sizes, seed)

    def compress_chunk_seeds(self, chunk_seeds_per_step, dkl_per_step):
        zipf_s_vals = []
        zipf_n_vals = []
        seeds = []

        for chunk_seeds, dkl in zip(chunk_seeds_per_step, dkl_per_step):
            chunk_sizes = get_chunk_sizes(dkl, self.max_chunk_size, self.chunk_padding)
            chunk_size_sum = sum(chunk_sizes)
            for chunk_seed, chunk_size in zip(chunk_seeds, chunk_sizes):
                zipf_n_vals.append(2 ** chunk_size)

                chunk_dkl = dkl * chunk_size / chunk_size_sum
                s = 1 + 1 / (chunk_dkl + np.exp(-1) * np.log(np.e + 1))
                zipf_s_vals.append(s)
                seeds.append(chunk_seed)

        return encode_zipf(zipf_s_vals, zipf_n_vals, seeds)

    def decompress_chunk_seeds(self, encoded_bytes, dkl_per_step):
        zipf_s_vals = []
        zipf_n_vals = []

        for dkl in dkl_per_step:
            chunk_sizes = get_chunk_sizes(dkl, self.max_chunk_size, self.chunk_padding)
            chunk_size_sum = sum(chunk_sizes)
            for chunk_size in chunk_sizes:
                zipf_n_vals.append(int(2 ** chunk_size))

                chunk_dkl = dkl * chunk_size / chunk_size_sum
                s = 1 + 1 / (chunk_dkl + np.exp(-1) * np.log(np.e + 1))
                zipf_s_vals.append(s)

        flattened_seeds = decode_zipf(zipf_s_vals, zipf_n_vals, encoded_bytes)
        chunk_seeds_per_step = []
        index = 0
        for dkl in dkl_per_step:
            chunk_sizes = get_chunk_sizes(dkl, self.max_chunk_size, self.chunk_padding)
            step_seeds = []
            for chunk_size in chunk_sizes:
                step_seeds.append(flattened_seeds[index])
                index += 1
            chunk_seeds_per_step.append(step_seeds)
        return chunk_seeds_per_step

