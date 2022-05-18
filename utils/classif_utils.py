import numpy as np

class MyLeaveNOut:
    """ Leave out N (# of distinct labels) paired blocks """
    def __init__(self, **kwargs):
        for k in kwargs:
            print(f"Received kwargs for {k} which will not be used")

    def split(self, X, y, groups):
        """ Return the indices of all splits

        Parameters
        ----------
        X : np.ndarray
            the data array, first dim to match up with the labels
        y : np.ndarray
            the labels array (1D)
        groups : iterable
            any iterable which matches up in length with the labels array.
            It should assign a unique group value to each entry in the
            labels vector.

        Returns
        -------
        splits : list[tuple(np.ndarray, np.ndarray)]
            list of tuples (i.e.) folds for cross validation. The first element
            of each tuple are the indices for the train data, the second
            element are the indices for testing

        """

        y = np.asarray(y)
        groups = np.asarray(groups)
        # get groups per label
        gm = {k: list(set(groups[y == k])) for k in set(y)}

        for v in gm.values():
            v.sort()

        # ensure that groups are distinct in labels
        assert all([set(s).intersection(groups[y != k]) == set()
                    for k, s in gm.items()]), " Groups are not unique in label"

        # ensure lengths match
        set_lens = [len(v) for v in gm.values()]
        if not all([e == set_lens[0] for e in set_lens]):
            print("Not all the same number of groups per label - "
                  "will zip and thus drop all groups longer than the min")

        Xidcs = np.arange(X.shape[0])

        splits = [
            (
             np.hstack([Xidcs[groups == list(gv)[j]] for gv in gm.values()
                        for j in range(min(set_lens)) if j != i]),                 # the non selected --> train set  # noqa
             np.hstack([Xidcs[groups == list(gv)[i]]
                        for gv in gm.values()]),   # the selected per grp --> test                                   # noqa
            )
            for i in range(min(set_lens))]

        return splits
