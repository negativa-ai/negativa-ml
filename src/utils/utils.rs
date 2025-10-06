use std::vec;

#[cfg(feature = "gpu")]

/// Get the compute capabilities of all available CUDA devices.
pub fn get_compute_capabilities() -> Vec<u32> {
    let dev_list = rust_gpu_tools::Device::all();
    let mut ccs = vec![];
    for dev in dev_list.iter() {
        let compute_capbility = dev.cuda_device().unwrap().compute_capability();
        ccs.push(compute_capbility.0 * 10 + compute_capbility.1);
    }
    ccs
}

#[cfg(not(feature = "gpu"))]
/// Default implementation when GPU feature is not enabled.
pub fn get_compute_capabilities() -> Vec<u32> {
    vec![]
}
