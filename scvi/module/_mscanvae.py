from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.distributions import Categorical, Normal
from torch.distributions import kl_divergence as kl
from torch.nn import functional as F

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.module.base import LossRecorder, auto_move_data
from scvi.nn import Decoder, Encoder

from ._classifier import Classifier
from ._multivae import MULTIVAE
from ._utils import broadcast_labels


class MSCANVAE(MULTIVAE):
    """
    Single-cell annotation using variational inference.

    This is an implementation of the scANVI model described in [Xu21]_,
    inspired from M1 + M2 model, as described in (https://arxiv.org/pdf/1406.5298.pdf).

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    y_prior
        If None, initialized to uniform probability over cell types
    labels_groups
        Label group designations
    use_labels_groups
        Whether to use the label groups
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    **vae_kwargs
        Keyword args for :class:`~scvi.module.VAE`
    """

    def __init__(
        self,
        n_input_genes: int = 0,
        n_input_regions: int = 0,
        n_input_proteins: int = 0,
        modality_weights: Literal["equal", "cell", "universal"] = "equal",
        modality_penalty: Literal["Jeffreys", "MMD", "None"] = "Jeffreys",
        n_batch: int = 0,
        n_obs: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        peak_likelihood: Literal["bernoulli", "poisson"] = "bernoulli",
        gene_dispersion: str = "gene",
        peak_dispersion: str = "peak",
        region_factors: bool = True,
        log_variational: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        latent_distribution: str = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,
        use_size_factor_key: bool = False,
        protein_background_prior_mean: Optional[np.ndarray] = None,
        protein_background_prior_scale: Optional[np.ndarray] = None,
        protein_dispersion: str = "protein",
        y_prior=None,
        labels_groups: Sequence[int] = None,
        use_labels_groups: bool = False,
        classifier_layers: int = 1,
        classifier_parameters: dict = dict(),
        **model_kwargs
    ):
        # log_variational=log_variational,
        super().__init__(
            n_input_genes=n_input_genes,
            n_input_regions=n_input_regions,
            n_input_proteins=n_input_proteins,
            modality_weights=modality_weights,
            modality_penalty=modality_penalty,
            n_batch=n_batch,
            n_obs=n_obs,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            region_factors=region_factors,
            gene_likelihood=gene_likelihood,
            gene_dispersion=gene_dispersion,
            peak_likelihood=peak_likelihood,
            peak_dispersion=peak_dispersion,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_size_factor_key=use_size_factor_key,
            latent_distribution=latent_distribution,
            deeply_inject_covariates=deeply_inject_covariates,
            encode_covariates=encode_covariates,
            protein_background_prior_mean=protein_background_prior_mean,
            protein_background_prior_scale=protein_background_prior_scale,
            protein_dispersion=protein_dispersion,
            **model_kwargs,
        )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        self.n_labels = n_labels
        # Classifier takes n_latent as input
        cls_parameters = {
            "n_layers": classifier_layers,
            "n_hidden": n_hidden,
            "dropout_rate": dropout_rate,
        }
        cls_parameters.update(classifier_parameters)
        self.classifier = Classifier(
            n_latent,
            n_labels=n_labels,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            **cls_parameters,
        )

        self.encoder_z2_z1 = Encoder(
            n_latent,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
        )

        self.decoder_z1_z2 = Decoder(
            n_latent,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

        self.y_prior = torch.nn.Parameter(
            y_prior
            if y_prior is not None
            else (1 / n_labels) * torch.ones(1, n_labels),
            requires_grad=False,
        )
        self.use_labels_groups = use_labels_groups
        self.labels_groups = (
            np.array(labels_groups) if labels_groups is not None else None
        )
        if self.use_labels_groups:
            if labels_groups is None:
                raise ValueError("Specify label groups")
            unique_groups = np.unique(self.labels_groups)
            self.n_groups = len(unique_groups)
            if not (unique_groups == np.arange(self.n_groups)).all():
                raise ValueError()
            self.classifier_groups = Classifier(
                n_latent, n_hidden, self.n_groups, classifier_layers, dropout_rate
            )
            self.groups_index = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.tensor(
                            (self.labels_groups == i).astype(np.uint8),
                            dtype=torch.uint8,
                        ),
                        requires_grad=False,
                    )
                    for i in range(self.n_groups)
                ]
            )

    @auto_move_data
    def classify(self, x, y, cell_idx, batch_index=None, cont_covs=None, cat_covs=None):

        # Obtain Representation #
        """
        if self.log_variational:
            x = torch.log(1 + x)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x, cont_covs), dim=-1)
        else:
            encoder_input = x
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        """
        # Get Data and Additional Covs
        x_rna, x_chr, x_pro, mask_expr, mask_acc, mask_pro = self.get_data(x, y)

        if cont_covs is not None and self.encode_covariates:
            encoder_input_expression = torch.cat((x_rna, cont_covs), dim=-1)
            encoder_input_accessibility = torch.cat((x_chr, cont_covs), dim=-1)
            encoder_input_protein = torch.cat((y, cont_covs), dim=-1)
        else:
            encoder_input_expression = x_rna
            encoder_input_accessibility = x_chr
            encoder_input_protein = y

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        # Z Encoders
        qzm_acc, qzv_acc, z_acc = self.z_encoder_accessibility(
            encoder_input_accessibility, batch_index, *categorical_input
        )
        qzm_expr, qzv_expr, z_expr = self.z_encoder_expression(
            encoder_input_expression, batch_index, *categorical_input
        )
        qzm_pro, qzv_pro, z_pro = self.z_encoder_protein(
            encoder_input_protein, batch_index, *categorical_input
        )

        ## mix representations
        if self.modality_weights == "cell":
            weights = self.mod_weights[cell_idx, :]
        else:
            weights = self.mod_weights.unsqueeze(0).expand(len(cell_idx), -1)

        qz_m = self.mix_modalities(
            (qzm_expr, qzm_acc, qzm_pro), (mask_expr, mask_acc, mask_pro), weights
        )
        # Obtain Representation #

        # Classify #
        # We classify using the inferred mean parameter of z_1 in the latent space
        z = qz_m
        if self.use_labels_groups:
            w_g = self.classifier_groups(z)
            unw_y = self.classifier(z)
            w_y = torch.zeros_like(unw_y)
            for i, group_index in enumerate(self.groups_index):
                unw_y_g = unw_y[:, group_index]
                w_y[:, group_index] = unw_y_g / (
                    unw_y_g.sum(dim=-1, keepdim=True) + 1e-8
                )
                w_y[:, group_index] *= w_g[:, [i]]
        else:
            w_y = self.classifier(z)
        # Classify #

        return w_y

    @auto_move_data
    def classification_loss(self, labelled_dataset):
        x = labelled_dataset[REGISTRY_KEYS.X_KEY]
        # y = labelled_dataset[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        cell_idx = labelled_dataset[REGISTRY_KEYS.INDICES_KEY].long().ravel()
        labels = labelled_dataset[REGISTRY_KEYS.LABELS_KEY]
        batch_idx = labelled_dataset[REGISTRY_KEYS.BATCH_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = (
            labelled_dataset[cont_key] if cont_key in labelled_dataset.keys() else None
        )

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = (
            labelled_dataset[cat_key] if cat_key in labelled_dataset.keys() else None
        )
        classification_loss = F.cross_entropy(
            self.classify(
                x,
                labels,
                cell_idx,
                batch_index=batch_idx,
                cat_covs=cat_covs,
                cont_covs=cont_covs,
            ),
            labels.view(-1).long(),
        )
        return classification_loss

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        feed_labels=False,
        kl_weight=1,
        labelled_tensors=None,
        classification_ratio=None,
    ):
        #  Get Variables #
        # Get the data
        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        x_rna, x_chr, x_pro, mask_expr, mask_acc, mask_pro = self.get_data(x, y)

        reconst_loss = self.get_reconstruction_loss(
            generative_outputs,
            inference_outputs,
            x_rna,
            x_chr,
            x_pro,
            mask_expr,
            mask_acc,
            mask_pro,
            x.shape[0],
            x.device,
        )

        qz1 = inference_outputs["qz"]
        z1 = inference_outputs["z"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        if feed_labels:
            label = tensors[REGISTRY_KEYS.LABELS_KEY]
        else:
            label = None
        is_labelled = False if label is None else True
        #  Get Variables #

        # Enumerate choices of label
        labels, z1s = broadcast_labels(label, z1, n_broadcast=self.n_labels)
        qz2, z2 = self.encoder_z2_z1(z1s, labels)
        pz1_m, pz1_v = self.decoder_z1_z2(z2, labels)

        # KL Divergence Z2
        mean = torch.zeros_like(qz2.loc)
        scale = torch.ones_like(qz2.scale)

        kl_divergence_z2 = kl(qz2, Normal(mean, scale)).sum(dim=1)
        # KL Divergence Z2

        # KL Divergence Z1
        loss_z1_unweight = -Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1s).sum(dim=-1)
        loss_z1_weight = qz1.log_prob(z1).sum(dim=-1)
        # KL Divergence Z1

        # KL Divergence L
        if not self.use_observed_lib_size:
            ql = inference_outputs["ql"]
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_divergence_l = kl(
                ql,
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_divergence_l = 0.0
        # KL Divergence L

        if is_labelled:
            loss = reconst_loss + loss_z1_weight + loss_z1_unweight
            kl_locals = {
                "kl_divergence_z2": kl_divergence_z2,
                "kl_divergence_l": kl_divergence_l,
            }
            if labelled_tensors is not None:
                classifier_loss = self.classification_loss(labelled_tensors)
                loss += classifier_loss * classification_ratio
                return LossRecorder(
                    loss,
                    reconst_loss,
                    kl_locals,
                    classification_loss=classifier_loss,
                    n_labelled_tensors=labelled_tensors[REGISTRY_KEYS.X_KEY].shape[0],
                )
            return LossRecorder(
                loss,
                reconst_loss,
                kl_locals,
                kl_global=torch.tensor(0.0),
            )

        probs = self.classifier(z1)
        reconst_loss += loss_z1_weight + (
            (loss_z1_unweight).view(self.n_labels, -1).t() * probs
        ).sum(dim=1)

        kl_divergence = (kl_divergence_z2.view(self.n_labels, -1).t() * probs).sum(
            dim=1
        )
        kl_divergence += kl(
            Categorical(probs=probs),
            Categorical(probs=self.y_prior.repeat(probs.size(0), 1)),
        )
        kl_divergence += kl_divergence_l

        loss = torch.mean(reconst_loss + kl_divergence * kl_weight)

        if labelled_tensors is not None:
            classifier_loss = self.classification_loss(labelled_tensors)
            loss += classifier_loss * classification_ratio
            return LossRecorder(
                loss,
                reconst_loss,
                kl_divergence,
                classification_loss=classifier_loss,
            )
        return LossRecorder(loss, reconst_loss, kl_divergence)
