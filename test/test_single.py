import time

import tsib


def test_single():
    """A seeded profile is reproducible: tsorb draws from the global numpy
    generator, so without the seed every call is a different realization and
    anything downstream of it (heat load, dispatch) drifts run to run."""

    starttime = time.time()

    first = tsib.simSingleHousehold(3, 2010, seed=1, get_hot_water=True, resample_mean=True)
    second = tsib.simSingleHousehold(3, 2010, seed=1, get_hot_water=True, resample_mean=True)

    print("Profile generation took " + str(time.time() - starttime))

    assert first.equals(second)


if __name__ == "__main__":
    test_single()
