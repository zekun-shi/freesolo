"""Guard the census boundary between wrapper calls and actual arithmetic."""
from experiments.rebuttal.native_cost import aggregate_common_kernels,forest_node_population
from experiments.rebuttal.custom_cost_formulas import cell_statistics,quantize_cells


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


def test_trimmed_forest_counts_only_complete_visible_chunks():
    record=dict(kind='build_forests_cuda',capacity=8,admitted=[8,3],
                selected_chunks=None,prefill_trim=6)
    assert forest_node_population(record)=={1:15,2:5,4:1}
    record['selected_chunks']=[[2,[1,None]],[0,[None,2]]]
    record['prefill_trim']=None
    assert forest_node_population(record)=={1:5,2:2,4:1}


def test_dirty_tasks_skip_incomplete_and_empty_chunks():
    record=dict(kind='lean_dirty_build_many',admitted=[8,3],tasks=[
        [0,0,0,3,0,1,0,0,-1,-1],
        [0,0,1,2,0,1,0,0,-1,-1],
        [0,0,0,0,0,0,0,0,-1,-1]])
    assert forest_node_population(record)=={1:8,2:4,4:2,8:1}


def test_tall_proposal_filter_and_subtrees_cover_each_node_once():
    proposals=[[1,0,0,10,0,1,0,0,-1,-1],
               [2,0,0,10,0,1,1,0,-1,-1],
               [2,0,0,10,0,1,1,1,-1,-1]]
    record=dict(kind='lean_dirty_build',admitted=[1024],new_admitted=[4],
                tasks=[],tall=[proposals,0,1])
    assert forest_node_population(record)=={1<<k:1024>>k for k in range(11)}
    record['new_admitted']=[0]
    assert forest_node_population(record)=={}


def test_slot_store_counts_masked_quantization_shape_and_both_kv_tensors():
    phase=dict(kernel_calls=[],observed_work={},slot_store_kernels=[
        dict(method='prefill',heads=2,width=64,size=32,flags=[0,0],
             launched_cells=8,retained_cells=3),
        dict(method='pool',heads=2,width=64,size=32,flags=[0,0],candidate_slots=72)])
    got=aggregate_common_kernels(phase)
    assert got['components']['slot_quantization']==quantize_cells(16,32,64)
    assert got['components']['slot_pool']==dict(flops=576,fp_comparisons=144)
