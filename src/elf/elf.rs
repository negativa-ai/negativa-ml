use elf::abi::{PF_R, PF_X, PT_LOAD};
use elf::endian::AnyEndian;
use elf::ElfBytes;

/// A struct to parse and manipulate 64-bit ELF files
/// TODO: now the elf struct only support 64 bit elf file, need to support 32 bit elf file if necessary
///
/// * `parsed_elf`: the parsed elf file
/// * `addr_offset_diff`: loaded memory addr - file offset
pub struct ELF64<'data> {
    parsed_elf: ElfBytes<'data, AnyEndian>, // the parsed elf file
    addr_offset_diff: i64,                  // loaded memory addr - file offset
}

impl<'data> ELF64<'data> {
    /// Create a new ELF64 struct from the given data
    pub fn new(data: &'data [u8]) -> ELF64<'data> {
        let parsed_elf = ElfBytes::<AnyEndian>::minimal_parse(data).unwrap();
        let executable_phdr = parsed_elf
            .segments()
            .unwrap()
            .iter()
            .find(|p| p.p_type == PT_LOAD && p.p_flags == PF_R | PF_X)
            .unwrap();
        let addr_offset_diff = executable_phdr.p_vaddr as i64 - executable_phdr.p_offset as i64;

        ELF64 {
            parsed_elf,
            addr_offset_diff,
        }
    }

    // get the underlying str of the symbol string from the .dynsym section
    fn get_dyn_symbol_bytes(&self, offset: usize) -> &'data [u8] {
        let _dyn_str_section_header = self
            .parsed_elf
            .section_header_by_name(".dynstr")
            .unwrap()
            .unwrap();
        let dyn_str = self
            .parsed_elf
            .section_data(&_dyn_str_section_header)
            .unwrap()
            .0;
        let mut end = offset;
        while dyn_str[end] != 0 {
            end += 1;
        }
        &dyn_str[offset..end]
    }

    // get the underlying str of the symbol string from the .symtab section
    fn get_debug_symbol_bytes(&self, offset: usize) -> &'data [u8] {
        let sym_str_section_header = self
            .parsed_elf
            .section_header_by_name(".strtab")
            .unwrap()
            .unwrap();
        let sym_str = self
            .parsed_elf
            .section_data(&sym_str_section_header)
            .unwrap()
            .0;
        let mut end = offset;
        while sym_str[end] != 0 {
            end += 1;
        }
        &sym_str[offset..end]
    }

    /// Get the file offset of the given symbol
    pub fn get_symbol_offset(&self, symbol_bytes: &[u8]) -> Option<u64> {
        match self.parsed_elf.symbol_table() {
            Ok(Some((symtab, _))) => {
                for (_, s) in symtab.iter().enumerate() {
                    let found_sym_bytes = self.get_debug_symbol_bytes(s.st_name as usize);
                    if symbol_bytes == found_sym_bytes {
                        return Some(s.st_value - self.addr_offset_diff as u64);
                    }
                }
            }
            _ => {}
        }

        match self.parsed_elf.dynamic_symbol_table() {
            Ok(Some((dynsym, _))) => {
                for (_, s) in dynsym.iter().enumerate() {
                    let found_sym_bytes = self.get_dyn_symbol_bytes(s.st_name as usize);
                    if symbol_bytes == found_sym_bytes {
                        return Some(s.st_value - self.addr_offset_diff as u64);
                    }
                }
            }
            _ => {}
        }

        None
    }

    /// Get the loaded memory address of the given symbol
    pub fn get_symbol_addr(&self, symbol_bytes: &[u8]) -> Option<u64> {
        let offset = self.get_symbol_offset(symbol_bytes);
        match offset {
            Some(off) => Some((off as i64 + self.addr_offset_diff) as u64),
            None => None,
        }
    }

    /// Get the file offset of the given section
    pub fn get_section_offset(&self, section_name: &str) -> Option<u64> {
        let section_header = self
            .parsed_elf
            .section_header_by_name(section_name)
            .unwrap();
        if section_header.is_none() {
            return None;
        }
        Some(section_header.unwrap().sh_offset)
    }

    /// Get the size of the given section
    pub fn get_section_size(&self, section_name: &str) -> Option<u64> {
        let section_header = self
            .parsed_elf
            .section_header_by_name(section_name)
            .unwrap();

        if section_header.is_none() {
            return None;
        }
        Some(section_header.unwrap().sh_size)
    }

    /// Check if the ELF file has GPU code section (.nv_fatbin)
    pub fn has_gpu_code(&self) -> bool {
        let section_header = self
            .parsed_elf
            .section_header_by_name(".nv_fatbin")
            .unwrap();
        section_header.is_some()
    }

    /// Get the file offset and size of the GPU code section (.nv_fatbin)
    pub fn get_gpu_code_offset(&self) -> Option<u64> {
        return self.get_section_offset(".nv_fatbin");
    }

    /// Get the size of the GPU code section (.nv_fatbin)
    pub fn get_gpu_code_size(&self) -> Option<u64> {
        return self.get_section_size(".nv_fatbin");
    }
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
    fn test_get_symbol_offset() {
        let so_path = fixture("libdemo.so");
        let data = std::fs::read(so_path.clone()).unwrap();
        let elf64 = ELF64::new(&data);
        let symbol = b"_Z6matMulPiS_S_iii";
        let offset = elf64.get_symbol_offset(symbol).unwrap();
        assert_eq!(offset, 0xabed);
    }

    #[test]
    fn test_get_symbol_addr() {
        let so_path = fixture("libdemo.so");
        let data = std::fs::read(so_path.clone()).unwrap();
        let elf64 = ELF64::new(&data);
        let symbol = b"_Z6matMulPiS_S_iii";
        let addr = elf64.get_symbol_addr(symbol).unwrap();
        assert_eq!(addr, 0xabed);
    }

    #[test]
    fn test_get_section_offset_size() {
        let so_path = fixture("libdemo.so");
        let data = std::fs::read(so_path.clone()).unwrap();
        let elf64 = ELF64::new(&data);

        let offset = elf64.get_section_offset(".text").unwrap();
        let size = elf64.get_section_size(".text").unwrap();

        assert_eq!(offset, 0xa9f0);
        assert_eq!(size, 0x7a132);
    }

    #[test]
    fn test_has_gpu_code() {
        let so_path = fixture("libdemo.so");
        let data = std::fs::read(so_path.clone()).unwrap();
        let elf64 = ELF64::new(&data);

        assert!(elf64.has_gpu_code());
    }

    #[test]
    fn test_get_gpu_code_offset_size() {
        let so_path = fixture("libdemo.so");
        let data = std::fs::read(so_path.clone()).unwrap();
        let elf64 = ELF64::new(&data);

        let offset = elf64.get_gpu_code_offset().unwrap();
        let size = elf64.get_gpu_code_size().unwrap();

        assert_eq!(offset, 0x948d0);
        assert_eq!(size, 0x63e0);
    }
}
