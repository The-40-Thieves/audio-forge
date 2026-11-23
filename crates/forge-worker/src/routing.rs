//! Backend routing for stem separation: local GPU worker, remote Colab worker, Rust DSP.
//!
//! Every attempt that is skipped or fails is recorded in `SeparationReport::fallbacks` and
//! logged to stderr, and the report names the backend that produced the stems.

use std::path::{Path, PathBuf};
use std::time::Duration;

use forge_core::{ForgeError, Result, StemBundle};
use serde::{Deserialize, Serialize};

use crate::local_gpu::{BUILD_SOURCE_ROOT, LocalWorkerCommand, resolve_worker_script};
use crate::{colab, dsp_split, local_gpu};

/// Default wall-clock limit for one local GPU worker run.
const DEFAULT_LOCAL_TIMEOUT: Duration = Duration::from_secs(900);
/// Default limit for one Colab request, upload and download included.
const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(900);

/// Execution backend selection for neural separation, restoration, and generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerBackend {
    /// Tries the local RTX GPU worker, then the Colab worker (if a URL is set), then Rust DSP.
    Auto,
    /// Uses the local ephemeral `uv` Python GPU worker, falling back to Rust DSP on failure.
    LocalGpu,
    /// Uses a remote Google Colab A100/L4 worker, falling back to Rust DSP on failure.
    RemoteColab { base_url: String },
    /// Pure-Rust zero-dependency spectral/Mid-Side stem separator (instant, 0 VRAM).
    PureRustDsp,
}

/// The backend that actually produced a `StemBundle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendUsed {
    LocalGpu,
    RemoteColab,
    PureRustDsp,
}

impl BackendUsed {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LocalGpu => "local_gpu",
            Self::RemoteColab => "remote_colab",
            Self::PureRustDsp => "pure_rust_dsp",
        }
    }
}

/// Stems plus the backend that made them and why earlier backends were passed over.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationReport {
    pub bundle: StemBundle,
    pub backend: BackendUsed,
    pub fallbacks: Vec<String>,
}

/// Quality tier for neural stem separation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeparationQuality {
    /// Single-pass separation (Mel-Band-RoFormer Kim Vocal or HTDemucs directly).
    Fast,
    /// Studio-grade BS-RoFormer Viperx 1297 (~12.98 dB SDR) vocal isolation + vocal-free instrumental cascade.
    #[default]
    Studio,
    /// Dual-RoFormer ensemble (BS-RoFormer 1297 + Mel-Band-RoFormer Kim Vocal) + vocal-free instrumental cascade.
    Ultra,
}

impl SeparationQuality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Studio => "studio",
            Self::Ultra => "ultra",
        }
    }
}

/// Parses a separation quality tier (`fast`, `studio`, `ultra`).
pub fn parse_quality(raw: &str) -> Result<SeparationQuality> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "fast" => Ok(SeparationQuality::Fast),
        "studio" => Ok(SeparationQuality::Studio),
        "ultra" => Ok(SeparationQuality::Ultra),
        other => Err(ForgeError::InvalidParameter(format!(
            "unknown separation quality '{other}' (expected fast, studio, or ultra)"
        ))),
    }
}

/// Everything routing needs. `from_env` reads the process environment; tests build it directly.
#[derive(Debug, Clone)]
pub struct SeparationConfig {
    pub backend: WorkerBackend,
    /// How to launch the local worker, or `None` when its script was not found.
    pub local_worker: Option<LocalWorkerCommand>,
    /// Colab endpoint that `Auto` tries after the local worker (`FORGE_COLAB_WORKER_URL`).
    pub colab_url: Option<String>,
    pub local_timeout: Duration,
    pub http_timeout: Duration,
    /// Fail instead of falling back to the DSP split when no neural backend produced stems
    /// (`FORGE_WORKER_REQUIRE_NEURAL`).
    pub require_neural: bool,
    /// Number of primary stems to separate (`2`, `4`, or `6`; default `2`).
    pub stems: u8,
    /// Neural separation quality tier (`fast`, `studio`, `ultra`; default `studio`).
    pub quality: SeparationQuality,
    /// Run BS-RoFormer de-reverb on the isolated vocal stem (default `true`).
    pub dereverb: bool,
    /// Also split isolated drums into Kick, Snare, Toms, Hi-Hat, and Cymbals (default `false`).
    pub drum_sep: bool,
    /// Also split isolated vocals into Lead Vocals and Backing Vocals / Harmonies via RoFormer Karaoke (default `false`).
    pub split_lead_backing: bool,
    /// Also split isolated vocals into Male and Female vocal stems (default `false`).
    pub split_male_female: bool,
    /// Also split isolated guitar into Electric Guitar and Acoustic Guitar (default `false`).
    pub split_guitar: bool,
    /// Also extract Strings, Brass, Woodwinds, and Synth from the residual Other stem (default `false`).
    pub split_orchestral: bool,
    /// Also extract live Crowd / Audience / Room Noise from the residual Other stem (default `false`).
    pub split_crowd: bool,
    /// Run optional Apollo / spectral bandwidth restoration post-pass on separated stems (default `false`).
    pub restore: bool,
}

impl Default for SeparationConfig {
    fn default() -> Self {
        Self {
            backend: WorkerBackend::Auto,
            local_worker: None,
            colab_url: None,
            local_timeout: DEFAULT_LOCAL_TIMEOUT,
            http_timeout: DEFAULT_HTTP_TIMEOUT,
            require_neural: false,
            stems: 2,
            quality: SeparationQuality::Studio,
            dereverb: true,
            drum_sep: false,
            split_lead_backing: false,
            split_male_female: false,
            split_guitar: false,
            split_orchestral: false,
            split_crowd: false,
            restore: false,
        }
    }
}

impl SeparationConfig {
    /// Reads `FORGE_WORKER_BACKEND` (`auto`, `local_gpu`, `colab`, `dsp`; default `auto`),
    /// `FORGE_COLAB_WORKER_URL`, `FORGE_GPU_WORKER_SCRIPT`, `FORGE_GPU_WORKER_TIMEOUT_SECS`
    /// `FORGE_COLAB_TIMEOUT_SECS` and `FORGE_WORKER_REQUIRE_NEURAL` (`1` or `true`).
    pub fn from_env() -> Self {
        let colab_url = env_nonempty("FORGE_COLAB_WORKER_URL");
        let script = resolve_worker_script(
            env_nonempty("FORGE_GPU_WORKER_SCRIPT").map(PathBuf::from),
            Some(Path::new(BUILD_SOURCE_ROOT)),
        );
        let backend = env_nonempty("FORGE_WORKER_BACKEND").map_or(WorkerBackend::Auto, |name| {
            parse_backend(&name, colab_url.as_deref()).unwrap_or_else(|e| {
                eprintln!("forge-worker: {e}; using auto");
                WorkerBackend::Auto
            })
        });
        Self {
            backend,
            local_worker: script.as_deref().map(LocalWorkerCommand::uv),
            colab_url,
            local_timeout: env_secs("FORGE_GPU_WORKER_TIMEOUT_SECS", DEFAULT_LOCAL_TIMEOUT),
            http_timeout: env_secs("FORGE_COLAB_TIMEOUT_SECS", DEFAULT_HTTP_TIMEOUT),
            require_neural: env_nonempty("FORGE_WORKER_REQUIRE_NEURAL")
                .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true")),
            stems: 2,
            quality: SeparationQuality::Studio,
            dereverb: true,
            drum_sep: false,
            split_lead_backing: false,
            split_male_female: false,
            split_guitar: false,
            split_orchestral: false,
            split_crowd: false,
            restore: false,
        }
    }

    /// Enables the full 23-slot hierarchical pro studio cascade in one call.
    pub fn enable_all_stems(&mut self) {
        self.stems = 6;
        self.dereverb = true;
        self.drum_sep = true;
        self.split_lead_backing = true;
        self.split_male_female = true;
        self.split_guitar = true;
        self.split_orchestral = true;
        self.split_crowd = true;
    }
}

/// Parses a `FORGE_WORKER_BACKEND` value. `colab` needs a URL.
pub fn parse_backend(name: &str, colab_url: Option<&str>) -> Result<WorkerBackend> {
    match name.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(WorkerBackend::Auto),
        "local_gpu" | "local" | "gpu" => Ok(WorkerBackend::LocalGpu),
        "dsp" | "pure_rust_dsp" | "rust" => Ok(WorkerBackend::PureRustDsp),
        "colab" | "remote_colab" => colab_url
            .map(|url| WorkerBackend::RemoteColab {
                base_url: url.to_string(),
            })
            .ok_or_else(|| {
                ForgeError::InvalidParameter(
                    "FORGE_WORKER_BACKEND=colab needs FORGE_COLAB_WORKER_URL".to_string(),
                )
            }),
        other => Err(ForgeError::InvalidParameter(format!(
            "unknown FORGE_WORKER_BACKEND '{other}' (expected auto, local_gpu, colab or dsp)"
        ))),
    }
}

fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn env_secs(key: &str, default: Duration) -> Duration {
    env_nonempty(key).map_or(default, |v| {
        parse_timeout_secs(&v).unwrap_or_else(|| {
            eprintln!(
                "forge-worker: ignoring {key}='{v}' (expected 0 < seconds <= {MAX_TIMEOUT_SECS})"
            );
            default
        })
    })
}

/// Longest accepted timeout, one week. Larger values (up to `inf`) would make
/// `Duration::from_secs_f64` panic or the deadline arithmetic overflow.
pub const MAX_TIMEOUT_SECS: f64 = 7.0 * 24.0 * 3600.0;

/// Parses a timeout in seconds, rejecting non-numbers, non-positive and out-of-range values.
pub fn parse_timeout_secs(value: &str) -> Option<Duration> {
    let secs = value.trim().parse::<f64>().ok()?;
    (secs > 0.0 && secs <= MAX_TIMEOUT_SECS).then(|| Duration::from_secs_f64(secs))
}

/// One neural backend to try before the DSP fallback.
enum Attempt<'a> {
    Local(&'a LocalWorkerCommand),
    Colab(&'a str),
}

/// Lists the neural attempts for `config`, recording skipped ones in `skipped`.
fn plan<'a>(config: &'a SeparationConfig, skipped: &mut Vec<String>) -> Vec<Attempt<'a>> {
    const NO_SCRIPT: &str = "local_gpu skipped: python-worker/forge_gpu_worker.py not found \
                             in the build source tree (set FORGE_GPU_WORKER_SCRIPT)";
    let mut local = || match &config.local_worker {
        Some(worker) => Some(Attempt::Local(worker)),
        None => {
            skipped.push(NO_SCRIPT.to_string());
            None
        }
    };
    match &config.backend {
        WorkerBackend::PureRustDsp => Vec::new(),
        WorkerBackend::LocalGpu => local().into_iter().collect(),
        WorkerBackend::RemoteColab { base_url } => vec![Attempt::Colab(base_url)],
        WorkerBackend::Auto => {
            let mut attempts: Vec<Attempt<'a>> = local().into_iter().collect();
            match config.colab_url.as_deref() {
                Some(url) => attempts.push(Attempt::Colab(url)),
                None => skipped
                    .push("remote_colab skipped: FORGE_COLAB_WORKER_URL is not set".to_string()),
            }
            attempts
        }
    }
}

impl Attempt<'_> {
    fn backend(&self) -> BackendUsed {
        match self {
            Self::Local(_) => BackendUsed::LocalGpu,
            Self::Colab(_) => BackendUsed::RemoteColab,
        }
    }

    fn run(
        &self,
        input: &Path,
        output_dir: &Path,
        config: &SeparationConfig,
    ) -> Result<StemBundle> {
        let stems = match config.stems {
            2 | 4 | 6 => config.stems,
            other => {
                return Err(ForgeError::InvalidParameter(format!(
                    "stems must be 2, 4, or 6, got {other}"
                )));
            }
        };
        let bundle = match self {
            Self::Local(worker) => local_gpu::separate(worker, input, output_dir, stems, config)?,
            Self::Colab(url) => colab::separate(url, input, output_dir, config.http_timeout)?,
        };
        if bundle.vocals.is_none() && bundle.other.is_none() {
            return Err(ForgeError::WorkerError(
                "worker returned neither a vocal nor an instrumental stem".to_string(),
            ));
        }
        Ok(bundle)
    }
}

/// Separates an audio file into a `StemBundle`. The local GPU and Colab workers return two
/// stems (`Vocals` and `Other`, the instrumental); the Pure-Rust Mid/Side + Linkwitz-Riley
/// fallback returns four (`Vocals`, `Drums`, `Bass`, `Other`).
pub fn separate_stems(
    input_path: &Path,
    output_dir: &Path,
    config: &SeparationConfig,
) -> Result<SeparationReport> {
    std::fs::create_dir_all(output_dir)?;
    let mut skipped = Vec::new();
    let mut fallbacks = Vec::new();
    for attempt in plan(config, &mut skipped) {
        let backend = attempt.backend();
        match attempt.run(input_path, output_dir, config) {
            Ok(bundle) => return Ok(finish(bundle, backend, fallbacks, skipped)),
            Err(e) => fallbacks.push(format!("{} failed: {e}", backend.as_str())),
        }
    }
    if config.require_neural {
        fallbacks.extend(skipped);
        return Err(ForgeError::WorkerError(format!(
            "no neural stem-separation backend succeeded and a neural backend is required: {}",
            if fallbacks.is_empty() {
                "the selected backend is pure_rust_dsp".to_string()
            } else {
                fallbacks.join("; ")
            }
        )));
    }
    let bundle = dsp_split::separate_stems_dsp(input_path, output_dir)?;
    Ok(finish(bundle, BackendUsed::PureRustDsp, fallbacks, skipped))
}

/// Logs and packages the result. Failures come first, then backends that were skipped.
fn finish(
    bundle: StemBundle,
    backend: BackendUsed,
    mut fallbacks: Vec<String>,
    skipped: Vec<String>,
) -> SeparationReport {
    fallbacks.extend(skipped);
    for reason in &fallbacks {
        eprintln!("forge-worker: {reason}");
    }
    eprintln!("forge-worker: stem separation ran on {}", backend.as_str());
    SeparationReport {
        bundle,
        backend,
        fallbacks,
    }
}

#[cfg(test)]
#[path = "routing_tests.rs"]
mod tests;
