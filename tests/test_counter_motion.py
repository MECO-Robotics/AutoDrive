import copy

import numpy as np
import pytest

from frc_defense.counter_motion import DRUM_NAME, configure_drum_motion, surface_velocity


PATH = ['test field', 'hub occurrence', 'counter occurrence', DRUM_NAME]


def manifest():
    return {'parts': [
        {'name': 'other', 'path': ['other'], 'axis': [1.,0.,0.], 'axis_point': [0.,0.,0.],
         'omega_per_surface_speed': 2.},
        {'name': DRUM_NAME, 'path': PATH[:], 'axis': [0.,1.,0.], 'axis_point': [0.,0.,0.],
         'omega_per_surface_speed': 1/.03175}], 'translation_m': [8.27024,4.03479,0.]}


def test_default_is_private_value_identical_and_identity_survives_reordering():
    source = manifest()
    assert configure_drum_motion(source, source_path=PATH) == source
    altered = configure_drum_motion(source, source_path=PATH, factor=-1)
    assert source == manifest()
    assert altered['parts'][0] == source['parts'][0]
    source['parts'].reverse()
    assert configure_drum_motion(source, source_path=PATH, factor=-1)['parts'][0] == altered['parts'][1]


def test_source_witness_tangents_and_unchanged_ramp():
    positive = manifest()['parts'][1]
    negative = configure_drum_motion(manifest(), source_path=PATH, factor=-1)['parts'][1]
    for point, desired in [([-.03175,0.,0.],[0.,0.,-1.]), ([0.,0.,-.03175],[1.,0.,0.])]:
        np.testing.assert_allclose(surface_velocity(positive,point,time_s=1.), -np.array(desired), atol=1e-15)
        np.testing.assert_allclose(surface_velocity(negative,point,time_s=1.), desired, atol=1e-15)
        np.testing.assert_allclose(surface_velocity(negative,point,time_s=.125), np.array(desired)/2, atol=1e-15)
        np.testing.assert_array_equal(surface_velocity(negative,point,time_s=0.), [0.,0.,0.])


@pytest.mark.parametrize('factor', [True, False, 0, 2, -2, 1., None, 'reverse'])
def test_invalid_factors_rejected(factor):
    with pytest.raises(ValueError):
        configure_drum_motion(manifest(), source_path=PATH, factor=factor)


@pytest.mark.parametrize('path', [[], 'not a path', ['missing'], ['other']])
def test_invalid_identity_rejected(path):
    with pytest.raises(ValueError):
        configure_drum_motion(manifest(), source_path=path, factor=-1)


def test_ambiguous_occurrence_rejected():
    source = manifest()
    source['parts'].append(copy.deepcopy(source['parts'][1]))
    with pytest.raises(ValueError):
        configure_drum_motion(source, source_path=PATH)


@pytest.mark.parametrize('field,value', [('axis',[0.,2.,0.]), ('axis_point',[float('nan'),0.,0.]),
                                      ('omega_per_surface_speed',float('inf'))])
def test_invalid_source_geometry_rejected(field, value):
    source = manifest()
    source['parts'][1][field] = value
    with pytest.raises(ValueError):
        configure_drum_motion(source, source_path=PATH)
