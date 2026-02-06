# -*- coding = utf-8 -*-
# @Time: 2025/10/18 14:21
# @Author: wisehone
# @File: CCE.py.py
# @SoftWare: PyCharm
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CCE(nn.Module):
    """"
    Channel Correlation Explore (CCE) Module
    division method maybe average or log~ or male~
    considering to expandable
    Input: X  R^{B*T×N} (T: sequence length, N: number of channels)
    Output:
        - X_transformed  R^{B*T×d} (channel-correlated sequence)
        - coherence_matrix  R^{N×N} (channel coherence matrix)
    """
    def __init__(self,
                 num_channels: int,
                 T: int,
                 bin_size: int = 5, # how many is clustered into a band 
                 k_sparse: int = 5,
                 use_laplacian: bool = False):
        super(CCE, self).__init__()

        self.num_channels = num_channels
        self.T = T
        self.num_bands = T // 2 // bin_size + 1
        self.k_parse = k_sparse
        self.use_laplacian = use_laplacian

        # Frequency band boundaries (uniform division) consider for another method
        self.register_buffer('band_boundaries',
                             torch.linspace(0, self.T // 2, self.num_bands, dtype=torch.long))

    def dft_tranform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DFT to each channel and extract first half due to conjugate symmetry
        :param x: in [B, N, T]
        :return: complex_sperctrum in [B, N, T // 2]
        """
        x_complex = torch.fft.rfft(x, dim=-1)
        return x_complex



    def frequency_band_aggregation(self, complex_spectrum: torch.Tensor) -> torch.Tensor:
        """
        Aggregate frequency bands and compress complex spectrum
        :param complex_spectrum:[B, N, T // 2]
        :return: complex_band_spectrum: [B, N, num_band]
        """
        B, N, _ = complex_spectrum.shape
        complex_band_spectrum = []

        # TODO: be more elegant
        for i in range(self.num_bands - 1):
            st_idx = self.band_boundaries[i]
            ed_idx = self.band_boundaries[i + 1]

            if st_idx == ed_idx:
                band_complex = complex_spectrum[:, :, st_idx:st_idx+1]
            else:
                band_complex = complex_spectrum[:, :, st_idx:ed_idx]

            # TODO: average or max?
            band_stat = torch.mean(band_complex, dim=-1)
            # band_avg = torch.mean(band_complex, dim=-1) # [B, N] 
            # band_max = torch.max(band_complex, dim=-1)

            # extract amplitude and phase
            amplitude = torch.abs(band_stat)
            phase = torch.angle(band_stat)

            # constract complex vector for this band
            z_band = amplitude * torch.exp(1j * phase)
            complex_band_spectrum.append(z_band)

        complex_band_spectrum = torch.stack(complex_band_spectrum, dim=-1)
        return complex_band_spectrum

    def complex_coherence_matrix(self, complex_band_spectrum: torch.Tensor) -> torch.Tensor:
        """
        Compute complex coherence matrix for batch
        :param complex_band_spectrum: [B, N, num_band]
        :return: coherence_matrix: [B, N, N]
        """
        B, N, num_band = complex_band_spectrum.shape

        # compute numerator [sum_b z_i(b)*z_j*(b)]
        z_conj = torch.conj(complex_band_spectrum) # [B, N, num_band]

        # use einsum for efficient computation
        # numerator[b, i, j] = |sum_k z[b, i, k] * z_conj[b, j, k]|
        numerator_complex = torch.einsum('bik,bjk->bij', complex_band_spectrum, z_conj)
        numerator_magnitude = torch.abs(numerator_complex)

        # compute denominator: sqrt(sum_b |z_i(b)| ^2 * sum_b |z_j(b)|^2)
        z_power = torch.sum(torch.abs(complex_band_spectrum) ** 2, dim=-1) # [B, N]

        # outer product for each batch
        denominator = torch.sqrt(
            z_power.unsqueeze(2) * z_power.unsqueeze(1)  # [B, N, N]
        )

        # avoid division by zero
        denominator = torch.clamp(denominator, min=1e-8)

        # coherence matrix
        coherence_matrix = numerator_magnitude / denominator # [B, N, N]
        return coherence_matrix

    def sparsify_coherence_matrix(self, coherence_matrix: torch.Tensor) -> torch.Tensor:
        """
        Sparsify coherence matrix with softmax, topk, and symmertrization
        :param coherence_matrix: [B, N, N]
        :return: sparse_coherence_matrix [B, N, N]
        """
        B, N, _ = coherence_matrix.shape

        # 1.row wise softmax
        A_softmax = F.softmax(coherence_matrix, dim=-1) # [B, N, N]

        # 2. topk
        topk_values, topk_indices = torch.topk(A_softmax, self.k_parse, dim=-1)

        # create mask for topk
        mask = torch.zeros_like(A_softmax)
        mask.scatter_(-1, topk_indices, 1.0) # write 1 to topk indices
        A_sparse = A_softmax * mask

        # 3. symmertrization
        A_symmetric = 0.5 * (A_sparse + A_sparse.transpose(1, 2))

        # 4. row normalization(frobnius norm)
        row_norms = torch.norm(A_symmetric, p='fro', dim=-1, keepdim=True) # [B, N, 1]
        row_norms = torch.clamp(row_norms, min=1e-8)
        A_normalized = A_symmetric / row_norms

        return A_normalized

    def laplacian_eigen_decomposition(self, coherence_matrix: torch.Tensor) -> torch.Tensor:
        """
        Perform Laplacian eigen decomposition for batch
        :param coherence_matrix: [B, N, N]
        :return: projection_matrix [B, N, N]
        """
        B, N, _ = coherence_matrix.shape

        # degree matrix D and normalized Laplacian
        degree = coherence_matrix.sum(dim=-1)  # [B, N]
        D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(degree + 1e-8))  # [B, N, N]
        I = torch.eye(N, device=coherence_matrix.device).expand(B, -1, -1)
        L = I - D_inv_sqrt @ coherence_matrix @ D_inv_sqrt  # [B, N, N]

        # batch eigen decomposition
        _, eigenvectors = torch.linalg.eigh(L)  # 支持batch
        W = eigenvectors[..., :N]  # [B, N, N]

        return W


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param x: [B, T, N]
        :return: x_transformed [B, T, N]
                cohernece_matrix [B, N, N]
        """
        B, T, N = x.shape

        # transpose to [B, N, T]
        if T == self.T and N == self.num_channels: # invert
            x_transposed = x.transpose(1, 2) # [B, N, T]
        else:
            x_transposed = x
        
        # 1.DFT
        complex_spectrum = self.dft_tranform(x_transposed) # [B, N, T // 2]

        # 2.Complex band spectrum
        complex_band_spectrum = self.frequency_band_aggregation(complex_spectrum) # [B, N, num_band]

        # 3.Complex coherence matrix
        coherence_matrix = self.complex_coherence_matrix(complex_band_spectrum) # [B, N, N]
        # 4.Sparsify coherence matrix
        sparse_coherence = self.sparsify_coherence_matrix(coherence_matrix) # [B, N, N]

        # 5.generate projection matrix
        if self.use_laplacian:
            projection_matrix = self.laplacian_eigen_decomposition(sparse_coherence)
        else:
            projection_matrix = sparse_coherence

        # 6.contain channel correlation
        # [B, N, N] @ [B, N, T] -> [B, N, T]
        x_transformed = torch.bmm(projection_matrix, x_transposed) # [B, N, T]

        # TODO: consider
        x_transformed = 0.5 * (x_transformed + x_transposed)

        return x_transformed, sparse_coherence
