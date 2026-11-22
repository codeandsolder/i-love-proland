// One invocation writes one unique, validated dense slot. Multiple
// disjoint updates share one transfer and compute dispatch.
struct CandidatePatch {
    slot: vec4<u32>,
    candidate: PackedInstance,
}
@group(0) @binding(0) var<storage,read_write> candidates:array<PackedInstance>;
@group(0) @binding(1) var<storage,read> patches:array<CandidatePatch>;
@group(0) @binding(2) var<uniform> count:u32;

@compute @workgroup_size(64)
fn patch_candidates(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x >= count {return;}
    let update_record=patches[id.x];
    candidates[update_record.slot.x]=update_record.candidate;
}
