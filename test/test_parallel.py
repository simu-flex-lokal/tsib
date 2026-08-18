import time

import pytest

import tsib


def test_parallel():

    starttime = time.time()

    res = tsib.simHouseholdsParallel(
        3, 2010, 2, cores=2, seeds=[1, 2], get_hot_water=True
    )

    print("Profile generation took " + str(time.time() - starttime))


def test_parallel_seeds_are_checked_against_the_household_count():
    """Seeds are matched to households by position, so a wrong count would
    silently give some household another household's profile."""
    with pytest.raises(ValueError):
        tsib.simHouseholdsParallel(3, 2010, 2, cores=2, seeds=[1])
