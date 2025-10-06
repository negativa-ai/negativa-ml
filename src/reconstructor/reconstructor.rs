use crate::locator::locator::ElementSpan;

/// Reconstructor is responsible for rewriting the shared object file based on the identified spans.
///
/// dst_so_path is the path to the destination shared object file to be rewritten, which is a copy of the original shared object file.
pub struct Reconstructor<'path> {
    dst_so_path: &'path str,
}

impl<'path> Reconstructor<'path> {
    /// Create a new Reconstructor instance.
    /// Copies the src_so_path to dst_so_path.
    pub fn new(src_so_path: &'path str, dst_so_path: &'path str) -> Self {
        // copy the src_so_path to dst_so_path
        std::fs::copy(src_so_path, dst_so_path).unwrap();
        Self { dst_so_path }
    }

    /// Rewrite the destination shared object file based on the provided spans.
    pub fn rewrite(&self, spans: &[ElementSpan]) {
        let mut so_data = std::fs::read(self.dst_so_path).unwrap();
        for span in spans.iter() {
            let start = span.start as usize;
            let end = span.end as usize;
            for i in start..end {
                so_data[i] = 0x01;
            }
        }
        std::fs::write(self.dst_so_path, so_data).unwrap();
    }
}
