import torch, random

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)


def stratified_sampling(indices, data, n_samples = 10, n_classes = 10) :
    """
    Sample data points with evenly distributed classes.
    Args:
        `indices`: indices of datapoints inside `data` to consider
        `data`: dataset of `(x, y)`-style tuples, where `y` is the target class label
        `n_samples`: total number of samples to return
        `n_classes`: number of classes, where class labels are assumed to be in `[0, ..., n_classes-1]`
    Returns:
        subset of indices of length n_samples, where each class is represented equally often
    """
    sampled = []
    bins = [[] for i in range(n_classes)]
    indices = iter(random.sample(indices, len(indices)))

    while n_samples > 0 :
        # Get datapoint index i and its class y
        i = next(indices)
        y = data[i][-1]
        bins[y].append(i)

        # Count non-empty bins
        n_nonempty = sum(map(bool, bins))
        # If no bins are empty...
        if n_nonempty in (n_classes, n_samples) :
            # ...collect one item per bin
            sampled.extend(b.pop(0) for b in bins if len(b) > 0)
            n_samples -= n_nonempty

    return sampled