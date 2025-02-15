"""File for computing log likelihood of the data."""
import torch


def compute_elbo(vae, data_loader, feed_labels=True, **kwargs):
    """
    Computes the ELBO.

    The ELBO is the reconstruction error + the KL divergences
    between the variational distributions and the priors.
    It differs from the marginal log likelihood.
    Specifically, it is a lower bound on the marginal log likelihood
    plus a term that is constant with respect to the variational distribution.
    It still gives good insights on the modeling of the data, and is fast to compute.
    """
    # Iterate once over the data and compute the elbo
    elbo = 0
    for tensors in data_loader:
        _, _, scvi_loss = vae(tensors, **kwargs)

        recon_loss = scvi_loss.reconstruction_loss
        kl_local = scvi_loss.kl_local
        elbo += torch.sum(recon_loss + kl_local).item()

    kl_global = scvi_loss.kl_global
    n_samples = len(data_loader.indices)
    elbo += kl_global
    return elbo / n_samples


# do each one
def compute_reconstruction_error(vae, data_loader, **kwargs):
    """
    Computes log p(x/z), which is the reconstruction error.

    Differs from the marginal log likelihood, but still gives good
    insights on the modeling of the data, and is fast to compute.
    """
    # Iterate once over the data and computes the reconstruction error
    log_lkl = {}
    for tensors in data_loader:
        loss_kwargs = dict(kl_weight=1)
        _, _, losses = vae(tensors, loss_kwargs=loss_kwargs)
        for key, value in losses._reconstruction_loss.items():
            if key in log_lkl:
                log_lkl[key] += torch.sum(value).item()
            else:
                log_lkl[key] = 0.0

    n_samples = len(data_loader.indices)
    for key, _ in log_lkl.items():
        log_lkl[key] = log_lkl[key] / n_samples
        log_lkl[key] = -log_lkl[key]
    return log_lkl
