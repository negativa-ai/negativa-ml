use log::debug;
use std::vec;

/// Represents the GPU code section containing multiple regions.
pub struct GPUCode {
    pub regions: Vec<Region>,
}

impl GPUCode {
    /// Create a new GPUCode instance by parsing the provided GPU code data.
    pub fn new(gpu_code_data: &[u8]) -> Self {
        let mut regions = Vec::new();
        let mut offset = 0;
        while offset < gpu_code_data.len() {
            let region = Region::new(gpu_code_data, offset as u64);
            offset += region.size() as usize;
            regions.push(region);
        }

        Self { regions }
    }
}

/// Represents a region within the GPU code section, containing multiple elements.
pub struct Region {
    pub header: RegionHeader,
    pub elements: Vec<Element>,
}

impl Region {
    /// Create a new Region instance by parsing the provided GPU code data starting from the specified offset.
    pub fn new(gpu_code_data: &[u8], start_offset: u64) -> Self {
        let region_data = &gpu_code_data[start_offset as usize..];
        let header = RegionHeader {
            header_size: u16::from_ne_bytes(region_data[6..8].try_into().unwrap()),
            fat_size: u64::from_ne_bytes(region_data[8..16].try_into().unwrap()),
        };
        let mut element_offset: usize = 16;
        let mut elements = vec![];
        while element_offset < header.fat_size as usize {
            debug!("Element offset: {}", element_offset);
            let element_header = ElementHeader {
                file_type: u16::from_ne_bytes(
                    region_data[element_offset..element_offset + 2]
                        .try_into()
                        .unwrap(),
                ),
                offset: u32::from_ne_bytes(
                    region_data[element_offset + 4..element_offset + 8]
                        .try_into()
                        .unwrap(),
                ),
                size: u64::from_ne_bytes(
                    region_data[element_offset + 8..element_offset + 16]
                        .try_into()
                        .unwrap(),
                ),
                capability: u32::from_ne_bytes(
                    region_data[element_offset + 28..element_offset + 32]
                        .try_into()
                        .unwrap(),
                ),
            };
            element_offset = element_header.offset as usize
                + element_offset as usize
                + element_header.size as usize;
            let element = Element {
                header: element_header,
            };
            elements.push(element);
        }

        Self { header, elements }
    }

    /// Calculate the total size of the region, including the header and FAT size.
    pub fn size(&self) -> u64 {
        self.header.fat_size + RegionHeader::size() as u64
    }

    // Find the most suitable capability that is less than or equal to the target capability.
    pub fn find_most_fit_capability(&self, target_cap: u32) -> u32 {
        let mut most_fit_cap = 0;
        for e in self.elements.iter() {
            if e.header.capability <= target_cap && e.header.capability > most_fit_cap {
                most_fit_cap = e.header.capability;
            }
        }
        most_fit_cap
    }
}

/// Represents an individual element within a region.
pub struct Element {
    pub header: ElementHeader,
}

/// Represents the header of a region, containing metadata about the region.
pub struct RegionHeader {
    pub header_size: u16,
    pub fat_size: u64,
}
impl RegionHeader {
    /// Get the size of the region header.
    pub fn size() -> u32 {
        16
    }
}

/// Represents the header of an element, containing metadata about the element.
pub struct ElementHeader {
    pub file_type: u16,
    pub offset: u32,
    pub size: u64,
    pub capability: u32,
}

#[cfg(test)]
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
    fn test_new_gpu_code() {
        let _ = env_logger::try_init();
        let so_path = fixture("libdemo.so");
        let data = std::fs::read(so_path).unwrap();
        let gpu_code_data = &data[0x948d0..0x9acb0];
        let gpu_code = GPUCode::new(gpu_code_data);
        let mut element_count = 0;
        let mut element_capabilities = vec![];
        let mut file_types = vec![];

        for region in gpu_code.regions.iter() {
            for element in region.elements.iter() {
                element_count += 1;
                element_capabilities.push(element.header.capability);
                file_types.push(element.header.file_type);
            }
        }

        assert_eq!(element_count, 4);
        assert_eq!(file_types, vec![2, 2, 2, 2]);
        assert_eq!(element_capabilities, vec![70, 75, 70, 75]);
    }

    #[test]
    fn test_find_most_fit_capability() {
        let _ = env_logger::try_init();
        let so_path = fixture("libdemo.so");
        let data = std::fs::read(so_path).unwrap();
        let gpu_code_data = &data[0x948d0..0x9acb0];
        let gpu_code = GPUCode::new(gpu_code_data);
        let region = &gpu_code.regions[0];

        let cap = region.find_most_fit_capability(72);
        assert_eq!(cap, 70);

        let cap = region.find_most_fit_capability(75);
        assert_eq!(cap, 75);

        let cap = region.find_most_fit_capability(80);
        assert_eq!(cap, 75);
    }
}
