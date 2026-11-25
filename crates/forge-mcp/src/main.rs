//! # `forge-mcp` — Model Context Protocol (MCP) Server
//!
//! Official `rmcp` 3.11 server (`2026-07-28` + backward-compatible `initialize`) exposing all
//! ForgeAudio DSP, music theory, vocal surgery, GPU worker, and REAPER 7.82 session tools,
//! resources, and prompts to AI agents (Antigravity, Claude Code, Cursor, REAPER AI scripts).
//!
//! File access is confined: every path argument must resolve inside the roots listed in
//! `FORGE_MCP_ROOT` (default: the directory the server was launched from). See [`sandbox`].

mod anchored;
mod external;
mod handlers;
mod resources;
mod sandbox;
#[cfg(all(test, unix))]
mod toctou_tests;
mod tools;

use clap::Parser;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as AutoServerBuilder;
use hyper_util::service::TowerToHyperService;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, GetPromptRequestParams, GetPromptResponse,
    Implementation, ListPromptsResult, ListResourcesResult, ListToolsResult,
    PaginatedRequestParams, ProtocolVersion, ReadResourceRequestParams, ReadResourceResponse,
    ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::io::stdio_pipes;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler, ServiceExt};
use sandbox::PathSandbox;
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

#[derive(Debug, Clone, PartialEq, Eq)]
enum TransportMode {
    Stdio,
    Http {
        addr: SocketAddr,
        allow_non_loopback: bool,
    },
}

#[derive(Debug, Clone, Parser)]
#[command(name = "forge-mcp", version, about = "ForgeAudio Studio MCP Server")]
struct McpCli {
    /// Serve Streamable HTTP MCP transport on <host:port> instead of stdio.
    #[arg(long, value_name = "HOST:PORT")]
    http: Option<SocketAddr>,

    /// Allow binding Streamable HTTP to non-loopback addresses (requires FORGE_MCP_HTTP_TOKEN).
    #[arg(long)]
    allow_non_loopback: bool,
}

fn cli_to_transport_mode(cli: McpCli) -> Result<TransportMode, String> {
    match cli.http {
        None => Ok(TransportMode::Stdio),
        Some(addr) => {
            if !addr.ip().is_loopback() && !cli.allow_non_loopback {
                return Err(format!(
                    "Refusing non-loopback HTTP bind address '{addr}' without --allow-non-loopback"
                ));
            }
            Ok(TransportMode::Http {
                addr,
                allow_non_loopback: cli.allow_non_loopback,
            })
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn parse_cli_args(args: &[String]) -> Result<TransportMode, String> {
    let cli =
        McpCli::try_parse_from(std::iter::once("forge-mcp").chain(args.iter().map(String::as_str)))
            .map_err(|e| e.to_string())?;
    cli_to_transport_mode(cli)
}

#[derive(Clone)]
struct ForgeMcpServer {
    sandbox: Arc<PathSandbox>,
    tools: Arc<Vec<Tool>>,
}

impl ForgeMcpServer {
    fn new(sandbox: Arc<PathSandbox>) -> Self {
        Self {
            sandbox,
            tools: Arc::new(tools::build_tools()),
        }
    }
}

impl ServerHandler for ForgeMcpServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .enable_prompts()
                .build(),
        )
        .with_protocol_version(ProtocolVersion::LATEST)
        .with_server_info(
            Implementation::new("forge-mcp", env!("CARGO_PKG_VERSION"))
                .with_title("ForgeAudio Studio MCP Server"),
        )
        .with_instructions(
            "Sandboxed ForgeAudio DSP, neural stem separation/restoration, music theory, and REAPER 7.82 session server. File paths must stay within FORGE_MCP_ROOT.",
        )
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items((*self.tools).clone()))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        self.tools.iter().find(|t| t.name.as_ref() == name).cloned()
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let name = request.name.into_owned();
        let args = Value::Object(request.arguments.unwrap_or_default());
        let sandbox = Arc::clone(&self.sandbox);
        tokio::task::spawn_blocking(move || handlers::call_tool_result(&name, &args, &sandbox))
            .await
            .map(CallToolResponse::from)
            .map_err(|e| McpError::internal_error(format!("Tool worker task failed: {e}"), None))
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult::with_all_items(
            resources::list_resources(),
        ))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, McpError> {
        resources::read_resource(&request.uri, &self.sandbox).map(ReadResourceResponse::from)
    }

    async fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, McpError> {
        Ok(ListPromptsResult::with_all_items(resources::list_prompts()))
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResponse, McpError> {
        resources::get_prompt(&request.name, request.arguments.as_ref())
            .map(GetPromptResponse::from)
    }
}

enum AuthBody<B> {
    Upstream(B),
    Unauthorized(&'static [u8]),
}

impl<B: hyper::body::Body + Unpin> hyper::body::Body for AuthBody<B>
where
    B::Data: From<&'static [u8]>,
{
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
        match &mut *self {
            Self::Upstream(inner) => std::pin::Pin::new(inner).poll_frame(cx),
            Self::Unauthorized(msg) => {
                if msg.is_empty() {
                    std::task::Poll::Ready(None)
                } else {
                    let slice = std::mem::take(msg);
                    std::task::Poll::Ready(Some(Ok(hyper::body::Frame::data(slice.into()))))
                }
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        match self {
            Self::Upstream(inner) => inner.is_end_stream(),
            Self::Unauthorized(msg) => msg.is_empty(),
        }
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        match self {
            Self::Upstream(inner) => inner.size_hint(),
            Self::Unauthorized(msg) => hyper::body::SizeHint::with_exact(msg.len() as u64),
        }
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= usize::from(x ^ y);
    }
    diff == 0
}

fn check_bearer_auth(headers: &hyper::HeaderMap, expected: &str) -> bool {
    let Some(raw) = headers
        .get(hyper::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    let Some((scheme, token)) = raw.trim().split_once(' ') else {
        return false;
    };
    scheme.eq_ignore_ascii_case("bearer")
        && constant_time_eq(token.trim().as_bytes(), expected.as_bytes())
}

fn resolve_http_policy(
    addr: SocketAddr,
    allow_non_loopback: bool,
    raw_secret: Option<&str>,
) -> Result<(bool, Option<String>), String> {
    let is_loopback = addr.ip().is_loopback();
    if !is_loopback && !allow_non_loopback {
        return Err(format!(
            "Refusing non-loopback HTTP bind address '{addr}' without --allow-non-loopback"
        ));
    }
    let secret = raw_secret
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string);
    if !is_loopback && secret.is_none() {
        return Err(
            "Refusing non-loopback HTTP bind without a non-empty FORGE_MCP_HTTP_TOKEN bearer secret"
                .to_string(),
        );
    }
    // Never disable Host-header checks on loopback addresses even if --allow-non-loopback is passed.
    let disable_host_check = !is_loopback && allow_non_loopback;
    Ok((disable_host_check, secret))
}

async fn run_http_on_listener(
    server: ForgeMcpServer,
    listener: TcpListener,
    mut config: StreamableHttpServerConfig,
    expected_secret: Option<Arc<str>>,
) {
    let ct = config.cancellation_token.clone();
    let mcp_svc = TowerToHyperService::new(StreamableHttpService::new(
        move || Ok(server.clone()),
        Arc::new(LocalSessionManager::default()),
        std::mem::take(&mut config),
    ));
    loop {
        tokio::select! {
            _ = ct.cancelled() => break,
            accepted = listener.accept() => {
                let Ok((stream, _)) = accepted else {
                    break;
                };
                let svc = mcp_svc.clone();
                let auth_header = expected_secret.clone();
                let conn_ct = ct.child_token();
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let hyper_svc = hyper::service::service_fn(
                        move |req: hyper::Request<hyper::body::Incoming>| {
                            let svc = svc.clone();
                            let auth_header = auth_header.clone();
                            async move {
                                if let Some(ref expected) = auth_header
                                    && !check_bearer_auth(req.headers(), expected)
                                {
                                    let resp = hyper::Response::builder()
                                        .status(hyper::StatusCode::UNAUTHORIZED)
                                        .header(hyper::header::WWW_AUTHENTICATE, "Bearer")
                                        .body(AuthBody::Unauthorized(b"Unauthorized\n"))
                                        .expect("valid 401 response");
                                    return Ok::<_, std::convert::Infallible>(resp);
                                }
                                let resp = hyper::service::Service::call(&svc, req)
                                    .await
                                    .expect("TowerToHyperService error is Infallible");
                                Ok::<_, std::convert::Infallible>(resp.map(AuthBody::Upstream))
                            }
                        },
                    );
                    let builder = AutoServerBuilder::new(TokioExecutor::new());
                    let conn = builder.serve_connection(io, hyper_svc);
                    tokio::select! {
                        res = conn => {
                            if let Err(err) = res {
                                eprintln!("forge-mcp: HTTP connection error: {err}");
                            }
                        }
                        _ = conn_ct.cancelled() => {}
                    }
                });
            }
        }
    }
}

async fn run_http(
    server: ForgeMcpServer,
    addr: SocketAddr,
    allow_non_loopback: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let raw_secret = std::env::var("FORGE_MCP_HTTP_TOKEN").ok();
    let (disable_host_check, secret) =
        resolve_http_policy(addr, allow_non_loopback, raw_secret.as_deref())
            .map_err(std::io::Error::other)?;
    let listener = TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;
    eprintln!("forge-mcp: Streamable HTTP listening on http://{local_addr}");
    let mut config = StreamableHttpServerConfig::default();
    if disable_host_check {
        config = config.disable_allowed_hosts();
    }
    let shutdown_ct = config.cancellation_token.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        shutdown_ct.cancel();
    });
    run_http_on_listener(server, listener, config, secret.map(Arc::<str>::from)).await;
    Ok(())
}

#[tokio::main]
async fn main() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(env_filter)
        .try_init();

    let cli = McpCli::parse();
    let mode = match cli_to_transport_mode(cli) {
        Ok(m) => m,
        Err(msg) => {
            eprintln!("forge-mcp: {msg}");
            std::process::exit(2);
        }
    };

    let sandbox = match PathSandbox::from_env() {
        Ok(sb) => Arc::new(sb),
        Err(e) => {
            eprintln!("forge-mcp: cannot set up the file sandbox: {e}");
            std::process::exit(1);
        }
    };
    let roots: Vec<String> = sandbox.roots().map(|r| r.display().to_string()).collect();
    eprintln!("forge-mcp: file access confined to {}", roots.join(", "));

    let server = ForgeMcpServer::new(sandbox);
    match mode {
        TransportMode::Stdio => match server.serve(stdio_pipes()).await {
            Ok(running) => {
                if let Err(e) = running.waiting().await {
                    eprintln!("forge-mcp: stdio server error: {e}");
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("forge-mcp: stdio initialize/discover failure: {e}");
                std::process::exit(1);
            }
        },
        TransportMode::Http {
            addr,
            allow_non_loopback,
        } => {
            if let Err(e) = run_http(server, addr, allow_non_loopback).await {
                eprintln!("forge-mcp: HTTP server failure: {e}");
                std::process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::{SessionSpec, build_session, install_session_jsfx, panic_message};
    use forge_core::{AudioBuffer, ForgeError};
    use serde_json::json;
    use std::panic;
    use std::path::Path;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    fn sandbox_in(dir: &Path) -> PathSandbox {
        PathSandbox::new([dir.to_path_buf()]).expect("sandbox")
    }

    fn write_test_mix(path: &Path, level: f64, sample_rate: u32) {
        forge_dsp::delivery::synthetic_delivery_fixture(level, sample_rate)
            .save_wav(path, 24)
            .expect("write fixture");
    }

    fn call_tool(sandbox: &PathSandbox, name: &str, arguments: Value) -> Value {
        let res = handlers::call_tool_result(name, &arguments, sandbox);
        serde_json::to_value(&res).expect("serialize CallToolResult")
    }

    fn is_error(result: &Value) -> bool {
        result["isError"].as_bool().expect("isError present")
    }

    fn text(result: &Value) -> &str {
        result["content"][0]["text"].as_str().expect("text content")
    }

    fn assert_tool_err_contains(sb: &PathSandbox, tool: &str, args: Value, needle: &str) {
        let res = call_tool(sb, tool, args);
        assert!(is_error(&res), "{tool} expected error: {}", text(&res));
        if !needle.is_empty() {
            assert!(
                text(&res).contains(needle),
                "{tool} expected '{needle}' in: {}",
                text(&res)
            );
        }
    }

    #[test]
    fn output_outside_root_is_rejected_before_any_write() {
        let root = tempfile::tempdir().expect("tempdir");
        let elsewhere = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("mix.wav"), 0.05, 48_000);

        let victim = elsewhere.path().join("victim.wav");
        assert_tool_err_contains(
            &sb,
            "forge_deliver",
            json!({ "input": "mix.wav", "output": victim.to_string_lossy() }),
            "outside the allowed roots",
        );
        assert!(!victim.exists());
        assert!(!victim.with_extension("delivery.json").exists());
    }

    fn setup_mix_sandbox(level: f64, sr: u32) -> (tempfile::TempDir, PathSandbox) {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("mix.wav"), level, sr);
        (root, sb)
    }

    fn parse_kv_args(spec: &str) -> Value {
        let mut map = serde_json::Map::new();
        for pair in spec.split(';') {
            if let Some((k, v)) = pair.split_once('=') {
                map.insert(k.to_string(), json!(v));
            }
        }
        Value::Object(map)
    }

    #[test]
    fn parent_dir_traversal_is_rejected_for_every_path_argument() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        for (tool, spec) in [
            ("forge_audit_audio", "path=../secret.wav"),
            ("forge_restore_and_defizz", "input=in.wav;output=../out.wav"),
            (
                "forge_spectral_unmask",
                "target=t.wav;sidechain=../s.wav;output=o.wav",
            ),
            (
                "forge_phase_align",
                "reference=../r.wav;target=t.wav;output=o.wav",
            ),
            (
                "forge_vocal_suite",
                "input=v.wav;output_dir=stems/../../escape",
            ),
            (
                "forge_music_theory_midi",
                "chords=Cmaj7;output_midi=../x.mid",
            ),
            (
                "forge_master_distrokid",
                "input=in.wav;output=o.wav;reference=../r.wav",
            ),
            (
                "forge_ingest_to_reaper",
                "input=in.wav;output_dir=../session",
            ),
            ("forge_neural_denoise", "input=../in.wav;output_dir=out"),
            (
                "forge_transcribe_midi",
                "input=in.wav;output_dir=../midi_out",
            ),
            (
                "forge_separate_stems",
                "input=in.wav;output_dir=../stems_out",
            ),
            ("forge_daw_inspect", "project=../session.forgeproj"),
            (
                "forge_daw_bounce",
                "project=session.forgeproj;output=../bounce.wav",
            ),
        ] {
            assert_tool_err_contains(&sb, tool, parse_kv_args(spec), "'..'");
        }
        assert!(!root.path().join("..").join("session").exists());
    }

    fn assert_mix_tool_error(tool: &str, args: Value, needle: &str, absent_rel: Option<&str>) {
        let (root, sb) = setup_mix_sandbox(0.05, 48_000);
        assert_tool_err_contains(&sb, tool, args, needle);
        if let Some(rel) = absent_rel {
            assert!(!root.path().join(rel).exists(), "{rel} should not exist");
        }
    }

    #[test]
    fn out_of_range_bpm_and_ceiling_are_tool_errors() {
        assert_mix_tool_error(
            "forge_music_theory_midi",
            json!({ "chords": "Dm9 G13", "output_midi": "x.mid", "bpm": 0 }),
            "bpm",
            Some("x.mid"),
        );
        assert_mix_tool_error(
            "forge_master_distrokid",
            json!({ "input": "mix.wav", "output": "master.wav", "ceiling_dbtp": 6.0 }),
            "ceiling_dbtp",
            Some("master.wav"),
        );
    }

    fn assert_delivery_ready(sample_rate: u32, output_rel: &str, expect_ready: bool) {
        let (root, sb) = setup_mix_sandbox(0.05, sample_rate);

        let result = call_tool(
            &sb,
            "forge_deliver",
            json!({ "input": "mix.wav", "output": output_rel }),
        );
        assert_eq!(
            is_error(&result),
            !expect_ready,
            "ready={expect_ready}: {}",
            text(&result)
        );
        assert_eq!(
            result["structuredContent"]["report"]["ready"],
            json!(expect_ready)
        );
        let body: Value = serde_json::from_str(text(&result)).expect("JSON body");
        assert_eq!(body["report"]["ready"], json!(expect_ready));
        assert!(root.path().join(output_rel).exists());
    }

    #[test]
    fn not_ready_delivery_reports_is_error() {
        assert_delivery_ready(22_050, "out/master.wav", false);
    }

    #[test]
    fn ready_delivery_reports_success() {
        assert_delivery_ready(48_000, "master.wav", true);
    }

    #[test]
    fn non_string_path_arguments_are_rejected() {
        assert_mix_tool_error(
            "forge_deliver",
            json!({ "input": "mix.wav", "output": "master.wav", "reference": 5 }),
            "'reference' must be a string path",
            Some("master.wav"),
        );
        assert_mix_tool_error(
            "forge_audit_audio",
            json!({ "path": ["mix.wav"] }),
            "'path' must be a string path",
            None,
        );

        let (_root, sb) = setup_mix_sandbox(0.05, 48_000);
        let result = call_tool(
            &sb,
            "forge_deliver",
            json!({ "input": "mix.wav", "output": "master.wav", "reference": null }),
        );
        assert!(!is_error(&result), "{}", text(&result));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_report_path_cannot_redirect_the_sidecar_write() {
        let root = tempfile::tempdir().expect("tempdir");
        let elsewhere = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("mix.wav"), 0.05, 48_000);
        let victim = elsewhere.path().join("victim.json");
        std::fs::write(&victim, "keep").expect("write victim");
        std::os::unix::fs::symlink(&victim, root.path().join("master.delivery.json"))
            .expect("symlink");

        let result = call_tool(
            &sb,
            "forge_deliver",
            json!({ "input": "mix.wav", "output": "master.wav" }),
        );
        assert!(is_error(&result));
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep");
        assert!(!root.path().join("master.wav").exists());
    }

    #[test]
    fn panic_payloads_are_reported() {
        let caught = panic::catch_unwind(|| panic!("boom {}", 42)).unwrap_err();
        assert_eq!(panic_message(caught.as_ref()), "boom 42");
        let caught = panic::catch_unwind(|| panic!("static")).unwrap_err();
        assert_eq!(panic_message(caught.as_ref()), "static");
    }

    fn repo_doc(relative: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(relative);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    }

    fn advertised_tool_names() -> Vec<String> {
        tools::build_tools()
            .into_iter()
            .map(|t| t.name.into_owned())
            .collect()
    }

    #[test]
    fn readme_tool_table_matches_the_server() {
        let readme = repo_doc("README.md");
        let section = readme
            .split("### MCP tools")
            .nth(1)
            .and_then(|rest| rest.split("\n### ").next())
            .expect("README.md has a '### MCP tools' section");
        let mut documented: Vec<String> = section
            .lines()
            .filter_map(|line| line.strip_prefix("| `"))
            .filter_map(|rest| rest.split('`').next())
            .map(str::to_string)
            .collect();
        let mut served = advertised_tool_names();
        documented.sort();
        served.sort();
        assert_eq!(
            documented, served,
            "README.md '### MCP tools' table is out of sync with build_tools()"
        );
    }

    #[test]
    fn architecture_guide_lists_every_tool() {
        let guide = repo_doc("docs/ARCHITECTURE_AND_ENGINEERING_GUIDE.md");
        let served = advertised_tool_names();
        let count = format!("exposing {} structured tools", served.len());
        assert!(
            guide.contains(&count),
            "architecture guide should say \"{count}\" to match build_tools()"
        );
        for name in &served {
            assert!(
                guide.contains(&format!("`{name}`")),
                "architecture guide does not mention MCP tool `{name}`"
            );
        }
    }

    #[test]
    fn pre_tag_render_is_removed_when_tagging_fails() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("mix.wav"), 0.05, 48_000);
        std::fs::create_dir(root.path().join("master.wav")).expect("mkdir");

        let result = call_tool(
            &sb,
            "forge_master_distrokid",
            json!({ "input": "mix.wav", "output": "master.wav", "isrc": "US-S1Z-26-00001" }),
        );
        assert!(is_error(&result), "{}", text(&result));
        assert!(!root.path().join("master.pre_tag.wav").exists());
    }

    #[test]
    fn session_jsfx_is_installed_where_the_rpp_references_it() {
        let resource = tempfile::tempdir().expect("tempdir");
        let report = install_session_jsfx(Ok(resource.path().to_path_buf()));
        assert_eq!(report["installed"], true, "{report}");

        let session = build_session(SessionSpec {
            bpm: 120.0,
            sample_rate: 48_000,
            duration_secs: 10.0,
            vocal_duration_secs: 10.0,
            drums_duration_secs: 10.0,
            bass_duration_secs: 10.0,
            other_duration_secs: 10.0,
            guitar_duration_secs: 10.0,
            keys_duration_secs: 10.0,
            vocal: Some(resource.path().join("vocals.wav")),
            drums: None,
            bass: None,
            other: None,
            guitar: None,
            keys: None,
            vocal_pitch_frames: Vec::new(),
            tempo_map: Vec::new(),
            chords_midi: resource.path().join("chords.mid"),
            drum_midi: resource.path().join("drums.mid"),
        });
        let effects = resource.path().join("Effects");
        let mut refs = 0;
        for line in session.build_rpp_string().lines() {
            let Some(rest) = line.trim_start().strip_prefix("<JS \"") else {
                continue;
            };
            let fx = &rest[..rest.find('"').expect("closing quote")];
            assert!(
                effects.join(fx).is_file(),
                "{fx} is not under {}",
                effects.display()
            );
            refs += 1;
        }
        assert_eq!(refs, 3);

        let missing = install_session_jsfx(Err(ForgeError::ReaperError("no REAPER".into())));
        assert_eq!(missing["installed"], false);
        assert!(
            missing["error"]
                .as_str()
                .expect("error")
                .contains("no REAPER")
        );
    }

    #[test]
    fn wave1_dsp_params_accepted_and_validated() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("mix.wav"), 0.08, 48_000);

        let res = call_tool(
            &sb,
            "forge_deliver",
            json!({
                "input": "mix.wav",
                "output": "out_punchy_16.wav",
                "target": "spotify",
                "limiter_style": "punchy",
                "channel_link": 0.5,
                "dither": "noise-shaped",
                "dither_bits": 16
            }),
        );
        assert!(!is_error(&res), "{}", text(&res));
        assert!(root.path().join("out_punchy_16.wav").is_file());

        assert_tool_err_contains(
            &sb,
            "forge_deliver",
            json!({ "input": "mix.wav", "output": "bad.wav", "limiter_style": "hyper_extreme" }),
            "limiter_style",
        );
        assert_tool_err_contains(
            &sb,
            "forge_deliver",
            json!({ "input": "mix.wav", "output": "bad.wav", "dither_bits": 4 }),
            "dither_bits",
        );

        let restored = call_tool(
            &sb,
            "forge_restore_and_defizz",
            json!({
                "input": "mix.wav",
                "output": "restored.wav",
                "tilt": -2.0,
                "stereo_link": 0.75,
                "mid_side": 0.5,
                "delta": true
            }),
        );
        assert!(!is_error(&restored), "{}", text(&restored));

        assert_tool_err_contains(
            &sb,
            "forge_restore_and_defizz",
            json!({ "input": "mix.wav", "output": "restored_bad.wav", "tilt": 12.0 }),
            "tilt",
        );

        let unmasked = call_tool(
            &sb,
            "forge_spectral_unmask",
            json!({
                "target": "mix.wav",
                "sidechain": "mix.wav",
                "output": "unmasked.wav",
                "mid_side": -0.5,
                "delta": true
            }),
        );
        assert!(!is_error(&unmasked), "{}", text(&unmasked));

        assert_tool_err_contains(
            &sb,
            "forge_neural_denoise",
            json!({ "input": "mix.wav", "output_dir": "denoised", "backend": "nonexistent" }),
            "backend",
        );
    }

    #[test]
    fn ai_suite_tools_sandbox_enforcement_and_ingest_features() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());

        let sr = 48_000_u32;
        let frames = (sr as usize) * 8;
        let mut ch: Vec<f64> = (0..frames)
            .map(|i| {
                let ts = (i as f64) / f64::from(sr);
                0.22 * (2.0 * std::f64::consts::PI * 220.0 * ts).sin()
            })
            .collect();
        let mut t = 0.2_f64;
        while t < 7.7 {
            let start = (t * f64::from(sr)).round() as usize;
            for k in 0..600 {
                if start + k < frames {
                    let env = (-(k as f64) / 100.0).exp();
                    ch[start + k] += 0.35
                        * env
                        * (2.0 * std::f64::consts::PI * 105.0 * (k as f64) / f64::from(sr)).sin();
                }
            }
            let bpm = if t < 4.0 { 100.0 } else { 132.0 };
            t += 60.0 / bpm;
        }
        AudioBuffer::from_channels(vec![ch.clone(), ch], sr)
            .unwrap()
            .save_wav(root.path().join("ai_song.wav"), 24)
            .unwrap();

        let kit_dir = root.path().join("kit");
        std::fs::create_dir_all(&kit_dir).unwrap();
        let kick: Vec<f64> = (0..1800)
            .map(|i| {
                0.8 * (-(i as f64) / 250.0).exp()
                    * (2.0 * std::f64::consts::PI * 65.0 * (i as f64) / f64::from(sr)).sin()
            })
            .collect();
        AudioBuffer::from_channels(vec![kick.clone(), kick], sr)
            .unwrap()
            .save_wav(kit_dir.join("kick.wav"), 24)
            .unwrap();

        let beats = call_tool(
            &sb,
            "forge_beat_track",
            json!({ "input": "ai_song.wav", "hint_bpm": 115.0, "backend": "rust" }),
        );
        assert!(!is_error(&beats), "{}", text(&beats));
        let beats_json: Value = serde_json::from_str(text(&beats)).unwrap();
        assert_eq!(beats_json["backend"], "pure_rust_onset");
        assert!(!beats_json["beats_secs"].as_array().unwrap().is_empty());

        let eval = call_tool(
            &sb,
            "forge_evaluate_quality",
            json!({ "input": "ai_song.wav", "reference": "ai_song.wav", "vocal": true, "backend": "rust" }),
        );
        assert!(!is_error(&eval), "{}", text(&eval));
        let eval_json: Value = serde_json::from_str(text(&eval)).unwrap();
        assert_eq!(eval_json["backend"], "rust-acoustic-proxy");

        for (tool, args) in [
            (
                "forge_neural_restore",
                json!({ "input": "../escape.wav", "output_dir": "out" }),
            ),
            (
                "forge_vocal_rescue",
                json!({ "input": "ai_song.wav", "output_dir": "out", "reference_voice": "../escape.wav" }),
            ),
            (
                "forge_evaluate_quality",
                json!({ "input": "ai_song.wav", "reference": "../escape.wav" }),
            ),
            (
                "forge_ingest_to_reaper",
                json!({ "input": "ai_song.wav", "output_dir": "sess", "drum_samples_dir": "../escape_kit" }),
            ),
        ] {
            assert_tool_err_contains(&sb, tool, args, "");
        }

        let delivered = call_tool(
            &sb,
            "forge_deliver",
            json!({
                "input": "ai_song.wav",
                "output": "ai_master.wav",
                "ai_track_mode": true,
                "hpss_shimmer": 0.4
            }),
        );
        assert!(!is_error(&delivered), "{}", text(&delivered));

        let ingested = call_tool(
            &sb,
            "forge_ingest_to_reaper",
            json!({
                "input": "ai_song.wav",
                "output_dir": "reaper_sess",
                "bpm": 115.0,
                "backend": "rust",
                "drum_samples_dir": "kit",
                "drum_replace_mix": 0.5
            }),
        );
        assert!(!is_error(&ingested), "{}", text(&ingested));
        let ingest_json: Value = serde_json::from_str(text(&ingested)).unwrap();
        assert!(ingest_json["beat_grid"]["beats_detected"].as_u64().unwrap() > 0);
        assert!(
            root.path()
                .join("reaper_sess/ForgeAudio_Session.rpp")
                .is_file()
        );
        let rpp_txt =
            std::fs::read_to_string(root.path().join("reaper_sess/ForgeAudio_Session.rpp"))
                .unwrap();
        if ingest_json["beat_grid"]["max_bpm_drift"]
            .as_f64()
            .unwrap_or(0.0)
            >= 1.5
        {
            assert!(
                rpp_txt.contains("<TEMPOENVEX"),
                "expected <TEMPOENVEX> in .rpp"
            );
        }
    }

    #[test]
    fn http_cli_bind_enforces_loopback_policy() {
        let to_vec = |xs: &[&str]| xs.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        assert_eq!(parse_cli_args(&[]).unwrap(), TransportMode::Stdio);
        assert!(matches!(
            parse_cli_args(&to_vec(&["--http", "127.0.0.1:8787"])).unwrap(),
            TransportMode::Http {
                allow_non_loopback: false,
                ..
            }
        ));
        assert!(matches!(
            parse_cli_args(&to_vec(&["--http", "[::1]:8787"])).unwrap(),
            TransportMode::Http {
                allow_non_loopback: false,
                ..
            }
        ));
        let rejected = parse_cli_args(&to_vec(&["--http", "0.0.0.0:8787"])).unwrap_err();
        assert!(rejected.contains("loopback"), "{rejected}");
        assert!(matches!(
            parse_cli_args(&to_vec(&["--http", "0.0.0.0:8787", "--allow-non-loopback"])).unwrap(),
            TransportMode::Http {
                allow_non_loopback: true,
                ..
            }
        ));
    }

    async fn rpc_roundtrip(
        writer: &mut tokio::io::WriteHalf<tokio::io::DuplexStream>,
        lines: &mut tokio::io::Lines<BufReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>>,
        msg: &Value,
    ) -> Value {
        let line = serde_json::to_string(msg).expect("serialize request");
        writer.write_all(line.as_bytes()).await.expect("write line");
        writer.write_all(b"\n").await.expect("write newline");
        writer.flush().await.expect("flush");
        let resp_line = lines.next_line().await.expect("read line").expect("eof");
        serde_json::from_str(&resp_line).expect("parse response")
    }

    fn meta_2026_07_28() -> Value {
        json!({
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientInfo": { "name": "test-client", "version": "1.0.0" },
            "io.modelcontextprotocol/clientCapabilities": {}
        })
    }

    type DuplexSession = (
        tempfile::TempDir,
        tokio::io::WriteHalf<tokio::io::DuplexStream>,
        tokio::io::Lines<BufReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>>,
        tokio::task::JoinHandle<()>,
    );

    fn spawn_duplex_session() -> DuplexSession {
        let (root, sb) = setup_mix_sandbox(0.05, 48_000);
        let server = ForgeMcpServer::new(Arc::new(sb));
        let (server_io, client_io) = tokio::io::duplex(64 * 1024);
        let (client_read, client_write) = tokio::io::split(client_io);
        let lines = BufReader::new(client_read).lines();
        let server_task = tokio::spawn(async move {
            let running = server.serve(server_io).await.expect("serve");
            running.waiting().await.expect("wait");
        });
        (root, client_write, lines, server_task)
    }

    #[tokio::test]
    async fn rmcp_2026_07_28_discover_tools_resources_and_prompts() {
        let (_root, mut client_write, mut lines, server_task) = spawn_duplex_session();

        let disc = rpc_roundtrip(
            &mut client_write,
            &mut lines,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "server/discover",
                "params": { "_meta": meta_2026_07_28() }
            }),
        )
        .await;
        let versions = disc["result"]["supportedVersions"].as_array().unwrap();
        assert!(versions.iter().any(|v| v == "2026-07-28"), "{disc}");

        let list = rpc_roundtrip(
            &mut client_write,
            &mut lines,
            &json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": { "_meta": meta_2026_07_28() }
            }),
        )
        .await;
        let listed_tools = list["result"]["tools"].as_array().unwrap();
        assert_eq!(listed_tools.len(), 18);
        assert!(listed_tools[0]["outputSchema"].is_object());
        assert!(listed_tools[0]["annotations"].is_object());
        assert!(list["result"]["ttlMs"].is_number());

        let call = rpc_roundtrip(
            &mut client_write,
            &mut lines,
            &json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "_meta": meta_2026_07_28(),
                    "name": "forge_audit_audio",
                    "arguments": { "path": "mix.wav" }
                }
            }),
        )
        .await;
        assert_eq!(call["result"]["resultType"], "complete");
        assert_eq!(call["result"]["isError"], false);
        assert!(call["result"]["structuredContent"]["integrated_lufs"].is_number());

        let res_list = rpc_roundtrip(
            &mut client_write,
            &mut lines,
            &json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "resources/list",
                "params": { "_meta": meta_2026_07_28() }
            }),
        )
        .await;
        assert_eq!(res_list["result"]["resources"].as_array().unwrap().len(), 3);

        for uri in ["forge://targets", "forge://models", "forge://roots"] {
            let read = rpc_roundtrip(
                &mut client_write,
                &mut lines,
                &json!({
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "resources/read",
                    "params": { "_meta": meta_2026_07_28(), "uri": uri }
                }),
            )
            .await;
            assert!(
                read["result"]["contents"][0]["text"]
                    .as_str()
                    .unwrap()
                    .starts_with('{'),
                "{uri}: {read}"
            );
        }

        let bad_res = rpc_roundtrip(
            &mut client_write,
            &mut lines,
            &json!({
                "jsonrpc": "2.0",
                "id": 6,
                "method": "resources/read",
                "params": { "_meta": meta_2026_07_28(), "uri": "forge://missing" }
            }),
        )
        .await;
        assert_eq!(bad_res["error"]["code"], json!(-32602));

        let prompts = rpc_roundtrip(
            &mut client_write,
            &mut lines,
            &json!({
                "jsonrpc": "2.0",
                "id": 7,
                "method": "prompts/list",
                "params": { "_meta": meta_2026_07_28() }
            }),
        )
        .await;
        assert_eq!(prompts["result"]["prompts"].as_array().unwrap().len(), 3);

        let prompt = rpc_roundtrip(
            &mut client_write,
            &mut lines,
            &json!({
                "jsonrpc": "2.0",
                "id": 8,
                "method": "prompts/get",
                "params": {
                    "_meta": meta_2026_07_28(),
                    "name": "master_ai_track",
                    "arguments": { "input_wav": "mix.wav", "target": "spotify" }
                }
            }),
        )
        .await;
        assert!(
            prompt["result"]["messages"][0]["content"]["text"]
                .as_str()
                .unwrap()
                .contains("mix.wav")
        );

        drop(client_write);
        drop(lines);
        server_task.await.expect("server shutdown");
    }

    #[tokio::test]
    async fn rmcp_legacy_initialize_versions_negotiate_cleanly() {
        for version in ["2025-11-25", "2024-11-05"] {
            let (_root, mut client_write, mut lines, server_task) = spawn_duplex_session();

            let init = rpc_roundtrip(
                &mut client_write,
                &mut lines,
                &json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": version,
                        "capabilities": {},
                        "clientInfo": { "name": "legacy-test", "version": "0.1" }
                    }
                }),
            )
            .await;
            assert_eq!(init["result"]["protocolVersion"], json!(version), "{init}");
            assert_eq!(init["result"]["serverInfo"]["name"], json!("forge-mcp"));

            let notif = serde_json::to_string(&json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }))
            .unwrap();
            client_write.write_all(notif.as_bytes()).await.unwrap();
            client_write.write_all(b"\n").await.unwrap();
            client_write.flush().await.unwrap();

            let list = rpc_roundtrip(
                &mut client_write,
                &mut lines,
                &json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
            )
            .await;
            assert!(list["result"].get("resultType").is_none(), "{list}");
            assert_eq!(list["result"]["tools"].as_array().unwrap().len(), 18);

            let call = rpc_roundtrip(
                &mut client_write,
                &mut lines,
                &json!({
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": { "name": "forge_audit_audio", "arguments": { "path": "mix.wav" } }
                }),
            )
            .await;
            assert!(call["result"].get("resultType").is_none(), "{call}");
            assert_eq!(call["result"]["isError"], json!(false));

            drop(client_write);
            drop(lines);
            server_task.await.expect("server shutdown");
        }
    }

    #[test]
    fn http_policy_enforces_token_on_non_loopback_and_keeps_loopback_host_check() {
        let loop_addr: SocketAddr = "127.0.0.1:8787".parse().unwrap();
        let any_addr: SocketAddr = "0.0.0.0:8787".parse().unwrap();

        let (disable_host, tok) = resolve_http_policy(loop_addr, true, None).unwrap();
        assert!(!disable_host, "loopback must keep Host-header validation");
        assert!(tok.is_none());

        assert!(resolve_http_policy(any_addr, false, Some("secret")).is_err());
        assert!(resolve_http_policy(any_addr, true, None).is_err());
        assert!(resolve_http_policy(any_addr, true, Some("   ")).is_err());

        let (disable_host, tok) = resolve_http_policy(any_addr, true, Some("my-secret")).unwrap();
        assert!(disable_host);
        assert_eq!(tok.as_deref(), Some("my-secret"));
    }

    #[tokio::test]
    async fn http_bearer_auth_and_loopback_host_check_enforce_access_control() {
        use tokio::io::AsyncReadExt;
        let (_root, sb) = setup_mix_sandbox(0.05, 48_000);
        let server = ForgeMcpServer::new(Arc::new(sb));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let local_addr = listener.local_addr().unwrap();
        let config = StreamableHttpServerConfig::default();
        let ct = config.cancellation_token.clone();
        let handle = tokio::spawn(async move {
            run_http_on_listener(server, listener, config, Some(Arc::from("s3cr3t"))).await;
        });

        async fn send_http(addr: SocketAddr, req_head: &str) -> String {
            let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
            stream.write_all(req_head.as_bytes()).await.unwrap();
            stream.flush().await.unwrap();
            let mut buf = Vec::new();
            stream.read_to_end(&mut buf).await.unwrap();
            String::from_utf8_lossy(&buf).into_owned()
        }

        let no_auth = send_http(
            local_addr,
            &format!("POST / HTTP/1.1\r\nHost: {local_addr}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"),
        )
        .await;
        assert!(no_auth.starts_with("HTTP/1.1 401 "), "{no_auth}");

        let wrong_auth = send_http(
            local_addr,
            &format!(
                "POST / HTTP/1.1\r\nHost: {local_addr}\r\nConnection: close\r\nAuthorization: Bearer wrong\r\nContent-Length: 0\r\n\r\n"
            ),
        )
        .await;
        assert!(wrong_auth.starts_with("HTTP/1.1 401 "), "{wrong_auth}");

        let bad_host = send_http(
            local_addr,
            "POST / HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\nAuthorization: Bearer s3cr3t\r\nContent-Length: 0\r\n\r\n",
        )
        .await;
        assert!(
            bad_host.starts_with("HTTP/1.1 403 ") || bad_host.starts_with("HTTP/1.1 421 "),
            "expected Host rejection on loopback, got: {bad_host}"
        );

        let valid_auth = send_http(
            local_addr,
            &format!(
                "POST / HTTP/1.1\r\nHost: {local_addr}\r\nConnection: close\r\nAuthorization: Bearer s3cr3t\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nContent-Length: 2\r\n\r\n{{}}"
            ),
        )
        .await;
        assert!(
            !valid_auth.starts_with("HTTP/1.1 401 ") && !valid_auth.starts_with("HTTP/1.1 403 "),
            "valid auth+host was unexpectedly rejected: {valid_auth}"
        );

        ct.cancel();
        let _ = handle.await;
    }

    #[test]
    fn quantize_stems_syncs_assets_and_session_uses_per_stem_sample_rate_duration() {
        let dir = tempfile::tempdir().expect("tempdir");
        // 44.1 kHz vocal & drum stems lasting exactly 2.0 seconds (88,200 frames)
        let sr = 44_100_u32;
        let frames = (sr as usize) * 2;
        let mut voc = vec![0.0_f64; frames];
        let mut drm = vec![0.0_f64; frames];
        for (i, s) in voc.iter_mut().enumerate() {
            let t = (i as f64) / f64::from(sr);
            if (1.15..=1.65).contains(&t) {
                *s = 0.45 * (2.0 * std::f64::consts::PI * 220.0 * t).sin();
            }
        }
        // Place a clear drum hit at t = 1.25 s (delayed beat 2 before quantize onto 120 BPM)
        let hit_smp = (1.25 * f64::from(sr)) as usize;
        for k in 0..400 {
            if hit_smp + k < frames {
                drm[hit_smp + k] = 0.85 * (-(k as f64) / 80.0).exp();
            }
        }
        let voc_buf = AudioBuffer::from_channels(vec![voc.clone(), voc], sr).expect("voc");
        let drm_buf = AudioBuffer::from_channels(vec![drm.clone(), drm], sr).expect("drm");
        assert!((handlers::buffer_duration_secs(&voc_buf) - 2.0).abs() < 1e-6);

        let scale = forge_vocal::VocalScale::parse("C major").expect("scale");
        let unquant_drum = forge_theory::extract_drum_midi_triggers(&drm_buf, 120.0, 480);
        let beats = [0.0, 0.55, 1.25, 1.85];
        let mut stems = vec![
            (dir.path().join("voc.wav"), voc_buf),
            (dir.path().join("drm.wav"), drm_buf),
        ];
        let (q_pitch, q_drums) = handlers::quantize_stems_and_sync_assets(
            &mut stems,
            Some(0),
            Some(1),
            &beats,
            120.0,
            &scale,
        )
        .expect("quantize");
        assert!(!q_pitch.is_empty());
        let (k_q, s_q, h_q) = q_drums.expect("drum midi present");
        let orig_ticks: Vec<u32> = unquant_drum
            .0
            .iter()
            .chain(&unquant_drum.1)
            .chain(&unquant_drum.2)
            .map(|n| n.start_tick)
            .collect();
        let quant_ticks: Vec<u32> = k_q
            .iter()
            .chain(&s_q)
            .chain(&h_q)
            .map(|n| n.start_tick)
            .collect();
        assert_ne!(
            orig_ticks, quant_ticks,
            "expected quantized MIDI ticks to shift"
        );

        let spec = SessionSpec {
            bpm: 120.0,
            sample_rate: 48_000,
            duration_secs: 2.0,
            vocal_duration_secs: 2.0,
            drums_duration_secs: 2.0,
            bass_duration_secs: 2.0,
            other_duration_secs: 2.0,
            guitar_duration_secs: 2.0,
            keys_duration_secs: 2.0,
            vocal: Some(dir.path().join("voc.wav")),
            drums: None,
            bass: None,
            other: None,
            guitar: None,
            keys: None,
            vocal_pitch_frames: q_pitch,
            tempo_map: Vec::new(),
            chords_midi: dir.path().join("chords.mid"),
            drum_midi: dir.path().join("drums.mid"),
        };
        let rpp = build_session(spec).build_rpp_string();
        assert!(
            rpp.contains("LENGTH 2.0000"),
            "expected 2.0 s LENGTH even at 48k session SR: {rpp}"
        );
    }
}
