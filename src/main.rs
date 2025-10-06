use clap::{Parser, Subcommand};
use log::{debug, info, warn};
use serde_json::json;
use std::env;

mod tracer;
use crate::elf::elf::ELF64;
use crate::locator::locator::KernelLocator;
use crate::tracer::tracer::{TraceReport, Tracer};
use crate::utils::utils::get_compute_capabilities;

mod elf;
mod locator;
mod reconstructor;
mod utils;

#[derive(Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Trace the workload to detect used kernels and loaded shared libraries
    Trace {
        /// System loader path, e.g., /usr/lib/x86_64-linux-gnu/ld-2.31.so
        #[arg(short, long, default_value = "/usr/lib/x86_64-linux-gnu/ld-2.31.so")]
        loader_path: String,

        /// Environment variables, if not set, reuse the current env
        #[arg(short, long, value_parser, num_args = 0.., value_delimiter = ' ')]
        env: Vec<String>,

        /// The file path to save the tracing report
        #[arg(short, long)]
        output: String,

        /// Cmd to run the workload, the executable must be the absolute path
        #[arg(trailing_var_arg = true)]
        cmd: Vec<String>,
    },

    /// Locate the unused device code segments in the loaded shared libraries, based on the output of the trace command
    Locate {
        /// Tracing report path, specified by --output in the trace command
        #[arg(short, long)]
        report_path: String,

        /// cuobjdump path, default to /usr/local/cuda/bin/cuobjdump
        #[arg(short, long, default_value = "/usr/local/cuda/bin/cuobjdump")]
        cuobjdump_path: String,

        /// Output dir to save the located unused device code segments
        #[arg(short, long)]
        output_dir: String,
    },

    /// Rewrite the unused device code segments to 0x1 in the shared libraries, based on the output of the locate command
    Reconstruct {
        /// A json file output by the locate command
        #[arg(short, long)]
        span_path: String,

        /// Output dir to save the reconstructed shared libraries
        #[arg(short, long)]
        output_dir: String, // Output dir
    },

    /// A convenient command to run trace and locate sequentially
    Debloat {
        /// System loader path, e.g., /usr/lib/x86_64-linux-gnu/ld-2.31.so
        #[arg(short, long, default_value = "/usr/lib/x86_64-linux-gnu/ld-2.31.so")]
        loader_path: String,

        /// Environment variables, if not set, reuse the current env
        #[arg(short, long, value_parser, num_args = 0.., value_delimiter = ' ')]
        env: Vec<String>,

        /// cuobjdump path, default to /usr/local/cuda/bin/cuobjdump
        #[arg(short, long, default_value = "/usr/local/cuda/bin/cuobjdump")]
        cuobjdump_path: String,

        /// Output dir to save the tracing report and located unused device code segments
        #[arg(short, long, default_value = "./nml_workspace")]
        output_dir: String,

        /// Cmd to run the workload, the executable must be the absolute path
        #[arg(trailing_var_arg = true)]
        cmd: Vec<String>,
    },
}

// Run the tracer
fn trace(loader_path: &str, env: &Vec<String>, cmd: &Vec<String>, output: &str) {
    let tracer = Tracer::new(&loader_path);
    let mut runtime_env = vec![];
    if env.len() == 0 {
        for (key, value) in env::vars() {
            runtime_env.push(format!("{}={}", key, value));
        }
        tracer.trace(cmd, &runtime_env, output);
    } else {
        tracer.trace(cmd, env, output);
    }
}

// Run the locator
fn locate(report_path: &str, cuobjdump_path: &str, output_dir: &str) {
    let report_file = std::fs::File::open(report_path).unwrap();
    let trace_report: TraceReport = serde_json::from_reader(report_file).unwrap();
    let loaded_sos = trace_report.loaded_sos;
    let detected_kernels = trace_report.detected_kernels;
    let compute_capabilities = get_compute_capabilities();
    if compute_capabilities.len() == 0 {
        warn!(
            "No GPU detected or GPU feature not enabled, skip locating unused device code segments"
        );
        return;
    }
    // TODO: support multi capabilities
    assert_eq!(compute_capabilities.len(), 1);
    let target_compute_capability = compute_capabilities[0];
    std::fs::create_dir_all(output_dir).unwrap();

    for so_path in loaded_sos.iter() {
        let so_data = std::fs::read(so_path).unwrap();
        let elf = ELF64::new(&so_data);
        if !elf.has_gpu_code() {
            continue;
        }
        let gpu_code_offset = elf.get_gpu_code_offset().unwrap();
        let gpu_code_size = elf.get_gpu_code_size().unwrap();
        let locator = KernelLocator::new(so_path, gpu_code_offset, gpu_code_size, cuobjdump_path);
        let spans =
            locator.locate_deletable_file_spans(&detected_kernels, target_compute_capability);
        let output_path = format!(
            "{}/{}.json",
            output_dir,
            so_path.split('/').last().unwrap().to_string()
        );
        let output_file = std::fs::File::create(output_path).unwrap();
        serde_json::to_writer_pretty(
            output_file,
            &json!({
                "so_path": so_path,
                "spans": spans
            }),
        )
        .unwrap();
    }
}

// Run the reconstructor
fn reconstruct(span_path: &str, output_dir: &str) {
    let span_file = std::fs::File::open(span_path).unwrap();
    let span_json: serde_json::Value = serde_json::from_reader(span_file).unwrap();
    debug!("Span json: {:?}", span_json);
    let so_path = span_json["so_path"].as_str().unwrap();
    let spans: Vec<locator::locator::ElementSpan> =
        serde_json::from_value(span_json["spans"].clone()).unwrap();
    std::fs::create_dir_all(output_dir).unwrap();
    let dst_so_path = format!(
        "{}/{}",
        output_dir,
        so_path.split('/').last().unwrap().to_string()
    );
    let reconstructor = reconstructor::reconstructor::Reconstructor::new(so_path, &dst_so_path);
    reconstructor.rewrite(&spans);
}

fn main() {
    env_logger::init();

    let args = Cli::parse();
    match args.command {
        Command::Trace {
            loader_path,
            env,
            cmd,
            output,
        } => {
            info!("Tracing report will be saved to: {}", output);
            trace(&loader_path, &env, &cmd, &output);
        }
        Command::Locate {
            report_path,
            cuobjdump_path,
            output_dir,
        } => {
            info!("Tracing report path: {}", report_path);
            info!("cuobjdump path: {}", cuobjdump_path);
            locate(&report_path, &cuobjdump_path, &output_dir);
        }
        Command::Reconstruct {
            span_path,
            output_dir,
        } => {
            info!("Span path: {}", span_path);
            info!("Reconstructed so will be saved to: {}", output_dir);
            reconstruct(&span_path, &output_dir);
        }
        Command::Debloat {
            loader_path,
            env,
            cuobjdump_path,
            output_dir,
            cmd,
        } => {
            // create output dir
            std::fs::create_dir_all(&output_dir).unwrap();

            let trace_output_file = format!("{}/trace.json", output_dir);
            trace(&loader_path, &env, &cmd, &trace_output_file);

            let span_path = format!("{}/spans", output_dir);
            locate(&trace_output_file, &cuobjdump_path, &span_path);
        }
    }
}
