# equivalent to torch.logsumexp(x, dim)
# overflow when exponentiating large values
def log_sum_exp(x, dim):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    m = torch.max(x, dim)[0]
    m2 = torch.max(x, dim, keepdims=True)[0]
    return m + torch.log(torch.sum(torch.exp(x-m2), dim))

def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a tensor.
    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n
    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                                 exp(values[i_1, ..., i_N])
            log( ------------------------------------------------------------ )
                    sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    return values - log_denominator


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a tensor.
    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n
    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    return torch.exp(lognormexp(values, dim=dim))
