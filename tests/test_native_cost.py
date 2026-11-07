"""Guard the census boundary between wrapper calls and actual arithmetic."""
from experiments.rebuttal.native_cost import aggregate_common_kernels
from experiments.rebuttal.custom_cost_formulas import cell_statistics


def call(name, inputs, repeats=1):
    return dict(function='test.'+name,inputs=inputs,calls=repeats)


def test_nested_cell_statistics_are_counted_once():
    phase=dict(kernel_calls=[
        call('cache_cell_stats',{},2),
        call('cell_stats',dict(K=dict(shape=[1,2,32,64]),M=1,s=32),2)],observed_work={})
    got=aggregate_common_kernels(phase)
    assert got['covered_kernel_arithmetic']==cell_statistics(4,32,64)
    assert got['wrappers_not_double_counted']=={'cache_cell_stats':2}
    assert got['complete_kernel_coverage']


def test_unhandled_forest_work_prevents_complete_total():
    phase=dict(kernel_calls=[call('forest_beam_pair',{},24)],observed_work={})
    got=aggregate_common_kernels(phase)
    assert got['pending_kernel_components']=={'forest_beam_pair':24}
    assert got['covered_kernel_arithmetic']=={}
    assert not got['complete_kernel_coverage']


def test_fsl_readout_does_not_hide_its_unresolved_selection():
    inputs=dict(q_t=dict(shape=[1,2,1,64]),out_short=None,pool=dict(empty=False),st=[32])
    phase=dict(kernel_calls=[call('lr_decode_read',dict(st=[32])),
                            call('lr_comb_int8',inputs)],observed_work={})
    got=aggregate_common_kernels(phase)
    assert got['components']['long_decode']['flops']>0
    assert got['pending_kernel_components']=={'lr_comb_int8.selection':1}
    assert got['wrappers_not_double_counted']=={'lr_decode_read':1}
    assert not got['complete_kernel_coverage']
