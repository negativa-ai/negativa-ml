use crate::elf::elf::ELF64;
use libc::{c_char, PTRACE_EVENT_CLONE, PTRACE_EVENT_EXEC, PTRACE_EVENT_FORK, PTRACE_EVENT_VFORK};
use log::{debug, info};
use nix::sys::ptrace::{self, AddressType};
use nix::sys::signal::Signal;
use nix::sys::wait::{waitpid, WaitStatus};
use nix::unistd::{fork, ForkResult, Pid};
use proc_maps::MapRange;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::io::{BufRead, BufReader, Write};
use std::os::raw::c_int;
use std::sync::mpsc::channel;
use std::sync::mpsc::Sender;
use std::{env, mem, thread, vec};
use tempfile::NamedTempFile;

const WORD_SIZE: usize = 8;

const RT_CONSISTENT: i32 = 0; /* Mapping change is complete.  */

/// Tracer is responsible for tracing the target process and its children to detect loaded shared libraries and used kernels.
pub struct Tracer {
    _dl_debug_state_addr: u64,
    _dl_debug_state_first_byte: u8,
    _r_debug_addr: u64,
    loader_path: String,
}

impl Tracer {
    /// Create a new Tracer instance.
    ///
    /// * `loader_path`: the path to the system loader, e.g., /usr/lib/x86_64-linux-gnu/ld-2.31.so
    pub fn new(loader_path: &str) -> Tracer {
        let data = std::fs::read(loader_path).expect("Failed to read loader");
        let _loader_elf = ELF64::new(&data);
        let _dl_debug_state_addr = _loader_elf
            .get_symbol_addr(b"_dl_debug_state")
            .expect("Failed to get _dl_debug_state offset");
        let _r_debug_addr = _loader_elf
            .get_symbol_addr(b"_r_debug")
            .expect("Failed to get _r_debug offset");

        let _dl_debug_state_offset = _loader_elf
            .get_symbol_offset(b"_dl_debug_state")
            .expect("Failed to get _dl_debug_state offset");
        let _dl_debug_state_first_byte = data[_dl_debug_state_offset as usize];

        Tracer {
            _dl_debug_state_addr,
            _dl_debug_state_first_byte,
            _r_debug_addr,
            loader_path: loader_path.to_string(),
        }
    }

    /// Trace the target process and its children to detect loaded shared libraries and used kernels.
    /// * `cmd`: the command to run the target process, the executable must be the absolute path
    /// * `env`: the environment variables to run the target process, if empty, reuse the current env
    /// * `output`: the file path to save the tracing report
    /// * return: the tracing report
    pub fn trace(&self, cmd: &[String], env: &[String], output: &str) -> TraceReport {
        let mut kernel_log_file = NamedTempFile::new().unwrap();
        match unsafe { fork() }.expect("Failed to fork") {
            ForkResult::Parent { child } => {
                waitpid(child, None).expect("wait child failed");
                ptrace::setoptions(
                    child,
                    ptrace::Options::PTRACE_O_TRACEFORK
                        | ptrace::Options::PTRACE_O_TRACECLONE
                        | ptrace::Options::PTRACE_O_TRACEVFORK
                        | ptrace::Options::PTRACE_O_TRACEEXEC,
                )
                .unwrap();

                let (so_sender, so_reciver) = channel::<String>();
                ptrace::detach(child, nix::sys::signal::Signal::SIGSTOP).expect("Fail to detach");
                Tracer::trace_multi_processes(
                    child,
                    self._dl_debug_state_addr as usize,
                    self._r_debug_addr as usize,
                    so_sender,
                    self.loader_path.clone(),
                );
                let mut loaded_sos = HashSet::new();
                for so_path in so_reciver {
                    if so_path.is_empty() {
                        continue;
                    }
                    // check if exists first and skip our kernel detector lib
                    let _so_path = std::path::Path::new(&so_path);
                    if !_so_path.exists() {
                        debug!("so_path: {} not exists, skipping...", so_path);
                        continue;
                    }
                    if so_path.contains("libkerneldetector.so") {
                        debug!("skipping kernel detector lib: {}", so_path);
                        continue;
                    }
                    // check if the path is absolute, if not, make it absolute
                    let abs_so_path = if _so_path.is_absolute() {
                        _so_path.canonicalize().unwrap().to_str().unwrap().to_string()
                    } else {
                        format!(
                            "{}/{}",
                            std::env::current_dir().unwrap().to_str().unwrap(),
                            so_path
                        )
                    };

                    loaded_sos.insert(abs_so_path);
                }
                info!("Tracing finished");

                // read the kernel log file to get the detected kernels
                let mut detected_kernels = HashSet::new();
                kernel_log_file.as_file_mut().flush().unwrap();
                let reader = BufReader::new(&kernel_log_file);

                for line in reader.lines() {
                    let line = line.unwrap().trim().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    detected_kernels.insert(line);
                }

                let kernel_report = json!(
                    {
                        "loaded_sos": loaded_sos,
                        "detected_kernels": detected_kernels,
                    }
                );

                let trace_report = TraceReport {
                    detected_kernels,
                    loaded_sos,
                };

                serde_json::to_writer_pretty(
                    std::fs::File::create(output).expect("Fail to create report file"),
                    &kernel_report,
                )
                .expect("Fail to write report file");

                trace_report
            }
            ForkResult::Child => {
                ptrace::traceme().expect("Fail to traceme in child");
                let path: &CStr = &CString::new(cmd[0].as_str()).unwrap();
                let mut env = env
                    .iter()
                    .map(|e| CString::new(e.clone()).unwrap())
                    .collect::<Vec<CString>>();

                env.push(
                    CString::new(
                        "KERNEL_LOGFILE=".to_string() + kernel_log_file.path().to_str().unwrap(),
                    )
                    .unwrap(),
                );
                env.push(
                    CString::new(format!(
                        "CUDA_INJECTION64_PATH={}/.negativa_ml/lib/libkerneldetector.so",
                        env::var("HOME").unwrap()
                    ))
                    .unwrap(),
                );

                let args = cmd
                    .iter()
                    .map(|e| CString::new(e.clone()).unwrap())
                    .collect::<Vec<CString>>();
                info!("Tracing started");
                debug!("envs in child: {:?}", env);
                nix::unistd::execve::<CString, CString>(path, &args, &env).unwrap();
                unreachable!();
            }
        }
    }

    fn trace_multi_processes(
        trace_pid: Pid,
        dl_debug_state_addr: usize,
        r_debug_addr: usize,
        so_sender: Sender<String>,
        loader_path: String,
    ) {
        thread::spawn(move || {
            debug!("Start tracing pid: {}", trace_pid);
            ptrace::attach(trace_pid).expect("Fail to attach");
            waitpid(trace_pid, None).expect("waitpid failed");
            ptrace::setoptions(
                trace_pid,
                ptrace::Options::PTRACE_O_TRACEFORK
                    | ptrace::Options::PTRACE_O_TRACECLONE
                    | ptrace::Options::PTRACE_O_TRACEVFORK
                    | ptrace::Options::PTRACE_O_TRACEEXEC,
            )
            .unwrap();

            let memory_maps: Vec<MapRange> =
                proc_maps::get_process_maps(trace_pid.as_raw() as proc_maps::Pid)
                    .expect("fail to get maps");

            let loader_base_addr = memory_maps
                .iter()
                .find(|m| {
                    m.filename().is_some()
                        && m.filename().unwrap().to_str().unwrap() == &loader_path
                })
                .expect("Fail to find base address")
                .start();
            let mut dl_debug_state_abs_addr = dl_debug_state_addr + loader_base_addr;
            let mut r_debug_abs_addr = r_debug_addr + loader_base_addr;
            Tracer::set_first_byte_at_addr(trace_pid, dl_debug_state_abs_addr, 0xcc);
            let mut cont_signal: Option<Signal> = None;
            loop {
                ptrace::cont(trace_pid, cont_signal).expect("Fail to cont");
                let status = waitpid(trace_pid, None).expect("waitpid failed");
                cont_signal = None;
                match status {
                    WaitStatus::Stopped(target_pid, nix::sys::signal::Signal::SIGTRAP) => {
                        debug!("Got SIGTRAP from pid: {}", target_pid);
                        let regs = ptrace::getregs(target_pid).expect("Fail to get regs");
                        let cur_addr = regs.rip - 1;
                        if cur_addr == dl_debug_state_abs_addr as u64 {
                            debug!(
                                "hit dl_debug_state_abs_addr: {:x?}, {}",
                                cur_addr, target_pid
                            );
                            let r_debug = Tracer::read_as::<RDebug>(target_pid, r_debug_abs_addr);
                            if r_debug.r_state == RT_CONSISTENT {
                                let mut link_map_addr = r_debug.r_map;
                                while link_map_addr != 0 as *const LinkMap {
                                    let link_map = Tracer::read_as::<LinkMap>(
                                        target_pid,
                                        link_map_addr as usize,
                                    );
                                    let so_path =
                                        Tracer::read_string(target_pid, link_map.l_name as usize);
                                    so_sender
                                        .send(so_path.clone())
                                        .expect("Fail to send so path");
                                    link_map_addr = link_map.l_next;
                                }
                            }
                        }
                    }
                    WaitStatus::PtraceEvent(
                        target_pid,
                        nix::sys::signal::Signal::SIGTRAP,
                        PTRACE_EVENT_CLONE,
                    )
                    | WaitStatus::PtraceEvent(
                        target_pid,
                        nix::sys::signal::Signal::SIGTRAP,
                        PTRACE_EVENT_FORK,
                    )
                    | WaitStatus::PtraceEvent(
                        target_pid,
                        nix::sys::signal::Signal::SIGTRAP,
                        PTRACE_EVENT_VFORK,
                    ) => {
                        let new_trace_pid = Pid::from_raw(
                            ptrace::getevent(target_pid).expect("Fail to getevent") as i32,
                        );
                        waitpid(new_trace_pid, None).expect("waitpid failed");
                        ptrace::detach(new_trace_pid, nix::sys::signal::Signal::SIGSTOP)
                            .expect("Fail to detach");
                        let new_so_sender = Sender::clone(&so_sender);
                        Tracer::trace_multi_processes(
                            new_trace_pid,
                            dl_debug_state_addr,
                            r_debug_addr,
                            new_so_sender,
                            loader_path.clone(),
                        );
                    }
                    WaitStatus::Stopped(_, signal) => {
                        cont_signal = Some(signal);
                    }
                    WaitStatus::Exited(traget_pid, exit_code) => {
                        debug!("Process {} exited with code {}", traget_pid, exit_code);
                        break;
                    }
                    WaitStatus::Signaled(traget_pid, Signal::SIGKILL, false) => {
                        debug!(
                            "Process {} killed by signal SIGKILL: {:?}",
                            traget_pid, status
                        );
                        break;
                    }
                    WaitStatus::Signaled(traget_pid, Signal::SIGTERM, false) => {
                        debug!(
                            "Process {} killed by signal SIGTERM: {:?}",
                            traget_pid, status
                        );
                        break;
                    }
                    WaitStatus::PtraceEvent(
                        target_pid,
                        nix::sys::signal::Signal::SIGTRAP,
                        PTRACE_EVENT_EXEC,
                    ) => {
                        debug!("PTRACE_EVENT_EXEC: {:?}", target_pid);
                        let memory_maps: Vec<MapRange> =
                            proc_maps::get_process_maps(target_pid.as_raw() as proc_maps::Pid)
                                .expect("fail to get maps");
                        debug!("Memory maps:  {}, {:x?}", target_pid, memory_maps);
                        let loader_base_addr = memory_maps
                            .iter()
                            .find(|m| {
                                m.filename().is_some()
                                    && m.filename().unwrap().to_str().unwrap() == &loader_path
                            })
                            .expect("Fail to find base address")
                            .start();
                        dl_debug_state_abs_addr = dl_debug_state_addr + loader_base_addr;
                        r_debug_abs_addr = r_debug_addr + loader_base_addr;
                        Tracer::set_first_byte_at_addr(trace_pid, dl_debug_state_abs_addr, 0xcc);
                    }
                    _ => {
                        debug!("Unhandled status: {:?}", status);
                    }
                }
            }
        });
    }

    fn set_first_byte_at_addr(pid: Pid, abs_addr: usize, first_byte: u8) {
        let orig_word = ptrace::read(pid, abs_addr as AddressType).unwrap();
        let word_to_write = (orig_word & !0xff) | first_byte as i64;
        unsafe {
            ptrace::write(pid, abs_addr as AddressType, word_to_write as AddressType)
                .expect("Fail to write");
        }
    }

    fn read_as<T>(pid: Pid, abs_addr: usize) -> T {
        let size = mem::size_of::<T>();
        let num_of_words = size / WORD_SIZE;
        assert_eq!(size % WORD_SIZE, 0);
        let mut words = vec![0i64; num_of_words];
        for i in 0..num_of_words {
            let addr = abs_addr + i * WORD_SIZE;
            let word = ptrace::read(pid, addr as AddressType).unwrap();
            words[i] = word;
        }
        let ptr = words.as_ptr();
        let t_ptr = ptr as *const T;
        unsafe { std::ptr::read::<T>(t_ptr) }
    }

    fn read_string(pid: Pid, address: usize) -> String {
        let mut str_bytes: Vec<u8> = Vec::new();
        let mut i = 0;
        loop {
            let word = ptrace::read(pid, (address + i) as AddressType).unwrap();
            str_bytes.push(word as u8);
            i += 1;
            if word == 0 {
                break;
            }
        }
        CStr::from_bytes_until_nul(&str_bytes)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string()
    }
}

// struct link_map
//   {
//     /* These first few members are part of the protocol with the debugger.
//        This is the same format used in SVR4.  */
//     ElfW(Addr) l_addr;		/* Difference between the address in the ELF
// 				   file and the addresses in memory.  */
//     char *l_name;		/* Absolute file name object was found in.  */
//     ElfW(Dyn) *l_ld;		/* Dynamic section of the shared object.  */
//     struct link_map *l_next, *l_prev; /* Chain of loaded objects.  */
//   };
#[repr(C)]
#[derive(Debug)]
struct LinkMap {
    l_addr: usize,
    l_name: *const c_char,
    l_ld: usize,
    l_next: *const LinkMap,
    l_prev: *const LinkMap,
}

#[repr(C)]
#[derive(Debug)]
struct RDebug {
    r_version: c_int,
    r_map: *const LinkMap,
    r_brk: usize,
    r_state: c_int,
    r_ldbase: usize,
}

#[derive(Serialize, Deserialize)]
pub struct TraceReport {
    pub detected_kernels: HashSet<String>,
    pub loaded_sos: HashSet<String>,
}
