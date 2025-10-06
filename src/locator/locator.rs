use super::gpu_code::GPUCode;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use tempfile::tempdir;

/// Locates deletable file spans in a shared object file based on detected GPU kernels and compute capability.
pub struct KernelLocator<'so_path> {
    so_path: &'so_path str,
    gpu_code: GPUCode,
    element_span: Vec<Vec<ElementSpan>>, // element_span[region_index][element_index] -> ElementSpan
    element_kernels: Vec<Vec<HashSet<String>>>, // element_kernels[region_index][element_index] -> kernel names
}

/// Represents the file span of an element within a region.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ElementSpan {
    pub start: u64, // start file offset (inclusive)
    pub end: u64,   // end file offset (exclusive)
}

const CUBLAS_INTERNAL_CONSTANT: &str = "_ZN6cublas8internal15deviceConstantsE";

impl<'so_path> KernelLocator<'so_path> {
    /// Create a new KernelLocator instance by parsing the provided shared object file and GPU code section.
    /// * `so_path`: Path to the shared object file.
    /// * `gpu_code_start_offset`: Start offset of the GPU code section within the shared object file.
    /// * `gpu_code_size`: Size of the GPU code section.
    /// * `cuobjdump_path`: Path to the cuobjdump executable.
    /// Returns a KernelLocator instance.
    pub fn new(
        so_path: &'so_path str,
        gpu_code_start_offset: u64,
        gpu_code_size: u64,
        cuobjdump_path: &str,
    ) -> KernelLocator<'so_path> {
        let so_data = std::fs::read(so_path).unwrap();
        let gpu_code_data = &so_data[gpu_code_start_offset as usize
            ..gpu_code_start_offset as usize + gpu_code_size as usize];
        let gpu_code = GPUCode::new(gpu_code_data);

        // extract all cubin paths
        let target_cubin_dir: tempfile::TempDir = tempdir().unwrap();
        let cubin_paths = Self::extract_all_cubins(
            so_path,
            target_cubin_dir.path().to_str().unwrap(),
            cuobjdump_path,
        );

        // calculate element spans and parse element kernel names
        let mut element_span = vec![];
        let mut offset = gpu_code_start_offset;

        let mut element_kernels = vec![];
        let mut cubin_path_index = 0;
        for region_idx in 0..gpu_code.regions.len() {
            let region = &gpu_code.regions[region_idx];
            let mut inner_offset = 0;
            let mut spans = vec![];
            let mut kernels = vec![];
            for element_idx in 0..region.elements.len() {
                // calculate element span
                let element = &region.elements[element_idx];
                let start = element.header.offset as u64
                    + inner_offset
                    + offset
                    + region.header.header_size as u64;
                let end = start + element.header.size;
                spans.push(ElementSpan { start, end });
                inner_offset += element.header.offset as u64 + element.header.size;

                // parse element kernel names
                if element.header.file_type != 2 {
                    // only process cubin file type
                    kernels.push(HashSet::new());
                } else {
                    let cubin_path = &cubin_paths[cubin_path_index];
                    cubin_path_index += 1;
                    let kernel_names = Self::extract_cubin_kernels(cubin_path, cuobjdump_path);
                    kernels.push(kernel_names);
                }
            }
            element_span.push(spans);
            element_kernels.push(kernels);
            let region_size = region.size();
            offset += region_size;
        }

        Self {
            so_path,
            gpu_code,
            element_span,
            element_kernels,
        }
    }

    /// Locate deletable file spans based on detected kernels and compute capability.
    /// * `detected_kernels`: Set of detected kernel names.
    /// * `compute_capability`: Target compute capability (e.g., 70 for sm_70).
    /// Returns a vector of ElementSpan representing deletable file spans.
    pub fn locate_deletable_file_spans(
        &self,
        detected_kernels: &HashSet<String>,
        compute_capability: u32,
    ) -> Vec<ElementSpan> {
        // given a set of detected kernels and compute capability, locate the file spans that can be deleted, i.e., no detected kernels in the spans
        let mut deletable_spans = vec![];
        for i in 0..self.gpu_code.regions.len() {
            let most_fit_cap =
                self.gpu_code.regions[i].find_most_fit_capability(compute_capability);
            for j in 0..self.gpu_code.regions[i].elements.len() {
                let element = &self.gpu_code.regions[i].elements[j];
                if element.header.capability != most_fit_cap {
                    deletable_spans.push(self.get_element_span(i, j).clone());
                } else {
                    if element.header.file_type != 2 {
                        continue;
                    }
                    let element_kernels = self.get_element_kernels(i, j);

                    let is_disjoint = detected_kernels.is_disjoint(element_kernels);
                    if is_disjoint {
                        // workaround: libcublas has some special internal constants needs to be retained
                        if self.so_path.contains("libcublas")
                            && element_kernels.contains(CUBLAS_INTERNAL_CONSTANT)
                        {
                            info!(
                                "Retaining libcublas internal constants, {}, {}, {}",
                                self.so_path, i, j
                            );
                            continue;
                        }
                        deletable_spans.push(self.get_element_span(i, j).clone());
                    }
                }
            }
        }
        deletable_spans
    }

    /// Get the file span of a specific element within a region.
    /// * `region_index`: Index of the region.
    /// * `element_index`: Index of the element within the region.
    /// Returns a reference to the ElementSpan.
    fn get_element_span(&self, region_index: usize, element_index: usize) -> &ElementSpan {
        if region_index >= self.gpu_code.regions.len()
            || element_index >= self.gpu_code.regions[region_index].elements.len()
        {
            panic!("region_index or element_index out of bounds");
        }
        &self.element_span[region_index][element_index]
    }

    /// Get the set of kernel names associated with a specific element within a region.
    /// * `region_index`: Index of the region.
    /// * `element_index`: Index of the element within the region.
    /// Returns a reference to the set of kernel names.
    fn get_element_kernels(&self, region_index: usize, element_index: usize) -> &HashSet<String> {
        if region_index >= self.gpu_code.regions.len()
            || element_index >= self.gpu_code.regions[region_index].elements.len()
        {
            panic!("region_index or element_index out of bounds");
        }

        &self.element_kernels[region_index][element_index]
    }

    /// Extract all cubin files from the given shared object file using cuobjdump.
    /// * `so_path`: Path to the shared object file.
    /// * `target_dir`: Directory to store the extracted cubin files.
    /// * `cuobjdump_path`: Path to the cuobjdump executable.
    /// Returns a vector of paths to the extracted cubin files.
    fn extract_all_cubins(so_path: &str, target_dir: &str, cuobjdump_path: &str) -> Vec<String> {
        debug!("Extracting cubins from {}", so_path);
        debug!("Target dir: {}", target_dir);

        let mut child = Command::new(cuobjdump_path)
            .current_dir(target_dir)
            .arg(so_path)
            .arg("-xelf")
            .arg("all") // Customize the path as needed
            .stdout(Stdio::piped())
            .spawn() // Capture stdout
            .expect("failed to execute command");
        let stdout = child.stdout.take().unwrap();
        let lines = BufReader::new(stdout).lines();
        let mut cubin_file_paths = Vec::new();
        for line in lines {
            // line is like "Extracting ELF file    1: libtorch_cuda.1.sm_50.cubin"
            let line = line.unwrap();
            debug!("entry: {:?}", line);
            let ele: Vec<&str> = line.split(":").collect();
            let filename = ele[1].trim();
            let path = std::path::Path::new(target_dir).join(filename);
            if path.is_file() {
                cubin_file_paths.push(path.to_str().unwrap().to_string());
            }
        }
        let exit_status = child.wait().unwrap();
        if !exit_status.success() {
            panic!(
                "cuobjdump failed with exit code: {:?}, {}",
                exit_status.code(),
                so_path
            );
        }

        debug!("Extracted cubins to {} done", target_dir);

        cubin_file_paths
    }

    /// Extract kernel names from a given cubin file using cuobjdump.
    /// * `cubin_path`: Path to the cubin file.
    /// * `cuobjdump_path`: Path to the cuobjdump executable.
    /// Returns a set of kernel names extracted from the cubin file.
    fn extract_cubin_kernels(cubin_path: &str, cuobjdump_path: &str) -> HashSet<String> {
        let mut output = Command::new(cuobjdump_path)
            .arg("-elf")
            .arg(cubin_path)
            .stdout(Stdio::piped()) // Capture stdout
            .spawn() // Start the command
            .expect("failed to execute command");

        let mut section_header_output = Vec::new();
        let mut symtable_output = Vec::new();
        // Use BufReader to read the output line by line
        if let Some(stdout) = output.stdout.take() {
            let reader = BufReader::new(stdout);
            let mut is_section_start = false;
            let mut is_symtab_start = false;
            // TODO: make the following parsing more robust and elegant
            for line in reader.lines() {
                match line {
                    Ok(line) => {
                        if is_section_start {
                            if line.trim() == "" {
                                is_section_start = false;
                            } else {
                                section_header_output.push(line);
                            }
                        } else if is_symtab_start {
                            if line.trim() == "" {
                                break;
                            } else {
                                symtable_output.push(line);
                            }
                        } else if line.trim() == "Sections:" {
                            is_section_start = true;
                        } else if line.trim() == ".section .symtab" {
                            is_symtab_start = true;
                            is_section_start = false;
                        }
                    }
                    Err(e) => eprintln!("Error reading line: {}", e),
                }
            }
        }

        let mut kernel_names = HashSet::new();
        for line in &section_header_output[1..] {
            let mut fields = line.split_whitespace();
            let sh_name = fields.nth(9).unwrap().to_string();
            if sh_name.starts_with(".text.") {
                kernel_names.insert(sh_name.strip_prefix(".text.").unwrap().to_string());
            }
        }

        // workaround for libcublas internal constants
        for line in &symtable_output[1..] {
            let mut fields = line.split_whitespace();

            let opt_st_name = fields.nth(6);
            let st_name = match opt_st_name {
                Some(name) => name.to_string(),
                None => {
                    warn!(
                        "Failed to parse symbol table line: {}, {}",
                        line, cubin_path
                    );
                    continue;
                }
            };

            if st_name == CUBLAS_INTERNAL_CONSTANT {
                kernel_names.insert(st_name);
                break;
            }
        }

        output.wait().unwrap();

        kernel_names
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    #[test]
    fn test_extract_all_cubins() {
        let _ = env_logger::try_init();
        let so_path = fixture("libdemo.so");
        let target_dir = tempdir().unwrap();
        let cuobjdump_path = "/usr/local/cuda/bin/cuobjdump";

        let cubin_paths = KernelLocator::extract_all_cubins(
            so_path.to_str().unwrap(),
            target_dir.path().to_str().unwrap(),
            cuobjdump_path,
        );

        assert_eq!(cubin_paths.len(), 4);
    }

    #[test]
    fn test_extract_cubin_kernels() {
        let _ = env_logger::try_init();
        let cubin_path = fixture("libdemo.3.sm_70.cubin");
        let cuobjdump_path = "/usr/local/cuda/bin/cuobjdump";

        let kernel_names =
            KernelLocator::extract_cubin_kernels(cubin_path.to_str().unwrap(), cuobjdump_path);

        assert_eq!(kernel_names.len(), 2);
        assert!(kernel_names.contains(&"_Z12matrixMulGPUPiS_S_iii".to_string()));
        assert!(kernel_names.contains(&"_Z16setScalarItemGPUiPiii".to_string()));
    }

    #[test]
    fn test_get_element_span() {
        let _ = env_logger::try_init();
        let so_path = fixture("libdemo.so");
        let cuobjdump_path = "/usr/local/cuda/bin/cuobjdump";
        let gpu_code_start_offset = 0x948d0;
        let gpu_code_size = 0x63e0;

        let locator = KernelLocator::new(
            so_path.to_str().unwrap(),
            gpu_code_start_offset,
            gpu_code_size,
            cuobjdump_path,
        );

        // region 0, element 0
        let span = locator.get_element_span(0, 0);
        assert_eq!((span.start, span.end), (0x94928, 0x94c90));

        // region 0, element 1
        let span = locator.get_element_span(0, 1);
        assert_eq!((span.start, span.end), (0x94cd8, 0x95040));

        // region 1, element 0
        let span = locator.get_element_span(1, 0);
        assert_eq!((span.start, span.end), (0x95098, 0x97f80));

        // region 1, element 1
        let span = locator.get_element_span(1, 1);
        assert_eq!((span.start, span.end), (0x97fc8, 0x9acb0));
    }

    #[test]
    fn get_element_kernels() {
        let _ = env_logger::try_init();
        let so_path = fixture("libdemo.so");
        let cuobjdump_path = "/usr/local/cuda/bin/cuobjdump";
        let gpu_code_start_offset = 0x948d0;
        let gpu_code_size = 0x63e0;

        let locator = KernelLocator::new(
            so_path.to_str().unwrap(),
            gpu_code_start_offset,
            gpu_code_size,
            cuobjdump_path,
        );

        let kernels = locator.get_element_kernels(0, 0);
        assert_eq!(kernels.len(), 0);

        let kernels = locator.get_element_kernels(0, 1);
        assert_eq!(kernels.len(), 0);

        let kernels = locator.get_element_kernels(1, 0);
        assert_eq!(kernels.len(), 2);
        assert!(kernels.contains(&"_Z12matrixMulGPUPiS_S_iii".to_string()));
        assert!(kernels.contains(&"_Z16setScalarItemGPUiPiii".to_string()));

        let kernels = locator.get_element_kernels(1, 1);
        assert_eq!(kernels.len(), 2);
        assert!(kernels.contains(&"_Z12matrixMulGPUPiS_S_iii".to_string()));
        assert!(kernels.contains(&"_Z16setScalarItemGPUiPiii".to_string()));
    }

    #[test]
    fn test_get_deletable_file_spans() {
        let _ = env_logger::try_init();
        let so_path = fixture("libdemo.so");
        let cuobjdump_path = "/usr/local/cuda/bin/cuobjdump";
        let gpu_code_start_offset = 0x948d0;
        let gpu_code_size = 0x63e0;
        let detected_kernels: HashSet<String> = vec!["_Z12matrixMulGPUPiS_S_iii"]
            .into_iter()
            .map(String::from)
            .collect();
        let locator = KernelLocator::new(
            so_path.to_str().unwrap(),
            gpu_code_start_offset,
            gpu_code_size,
            cuobjdump_path,
        );

        let deletable_spans = locator.locate_deletable_file_spans(&detected_kernels, 75);

        assert_eq!(deletable_spans.len(), 3);
        assert_eq!(
            (deletable_spans[0].start, deletable_spans[0].end),
            (0x94928, 0x94c90)
        );
        assert_eq!(
            (deletable_spans[1].start, deletable_spans[1].end),
            (0x94cd8, 0x95040)
        );
        assert_eq!(
            (deletable_spans[2].start, deletable_spans[2].end),
            (0x95098, 0x97f80)
        );

        let deletable_spans = locator.locate_deletable_file_spans(&detected_kernels, 70);
        assert_eq!(deletable_spans.len(), 3);
        assert_eq!(
            (deletable_spans[0].start, deletable_spans[0].end),
            (0x94928, 0x94c90)
        );
        assert_eq!(
            (deletable_spans[1].start, deletable_spans[1].end),
            (0x94cd8, 0x95040)
        );
        assert_eq!(
            (deletable_spans[2].start, deletable_spans[2].end),
            (0x97fc8, 0x9acb0)
        );
    }
}
