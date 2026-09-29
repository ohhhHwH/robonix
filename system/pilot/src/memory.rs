// SPDX-License-Identifier: MulanPSL-2.0
// Long-term-memory dispatch (best-effort, fire-and-forget on errors).
//
// Plan memory uses standalone `ptdl_store.json` (via `ptdl_remember` /
// `ptdl_retrieve` MCP tools) — independent of the CKG graph_store.
//
// `prefetch` runs before the first VLM call (opt-in via
//  ROBONIX_MEMORY_PREFETCH_ENABLED=1).
// `save_plan` runs after a successful plan completes (fire-and-forget).
// `try_compact` on session_end.
//
// Missing providers are silently tolerated — memory is never load-bearing.

use crate::pb::contracts::robonix_system_executor_execute_client::RobonixSystemExecutorExecuteClient;
use crate::pb::executor::rtdl_event::RtdlEventEnum;
use crate::pb::pilot::rtdl_node_state::RtdlNodeStateEnum;
use crate::pb::pilot::{CapabilityCall, CapabilityCallResult, Plan, RtdlNode};
use crate::planner::{ExecutorConn, TreeStep};
use robonix_atlas::client::AtlasClient;
use robonix_scribe::{debug, info, warn};
use tonic::Request;
use uuid::Uuid;

const RTDL_SEQUENCE: u32 = 0;
const RTDL_PARALLEL: u32 = 1;
const RTDL_DO: u32 = 2;

fn single_call_plan(
    plan_id: String,
    session_id: String,
    round: u32,
    call: CapabilityCall,
    label: &str,
) -> Plan {
    Plan {
        plan_id,
        session_id,
        round,
        nodes: vec![
            RtdlNode {
                node_kind: RTDL_SEQUENCE,
                children: vec![1],
                call: None,
                op_id: format!("{label}_sequence"),
                description: format!("{label} dispatch wrapper"),
            },
            RtdlNode {
                node_kind: RTDL_DO,
                children: Vec::new(),
                call: Some(call),
                op_id: format!("{label}_call"),
                description: format!("{label} dispatch"),
            },
        ],
        root_index: 0,
    }
}

/// Reconstruct a `Plan` proto from the JSON produced by
/// `planner::plan_to_json` (the saved `rtdl_plan`), assigning a fresh
/// `plan_id` / `session_id` for the replay dispatch. Best-effort: any parse or
/// shape mismatch returns `None` so replay falls back to normal planning.
fn json_to_plan(json: &str, plan_id: String, session_id: String) -> Option<Plan> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let root_index = v.get("root_index")?.as_u64()? as u32;
    let mut nodes = Vec::new();
    for node in v.get("nodes")?.as_array()? {
        let node_kind = match node.get("node_kind")?.as_str()? {
            "sequence" => RTDL_SEQUENCE,
            "parallel" => RTDL_PARALLEL,
            "do" => RTDL_DO,
            _ => return None,
        };
        let children = node
            .get("children")
            .and_then(|c| c.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as u32))
                    .collect()
            })
            .unwrap_or_default();
        let call = match node.get("call") {
            Some(serde_json::Value::Object(c)) => {
                let args_json = c
                    .get("args")
                    .map(|args| serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string()))
                    .unwrap_or_else(|| "{}".to_string());
                Some(CapabilityCall {
                    call_id: c
                        .get("call_id")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string(),
                    provider_id: c
                        .get("provider_id")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string(),
                    contract_id: c
                        .get("contract_id")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string(),
                    args_json,
                })
            }
            _ => None,
        };
        nodes.push(RtdlNode {
            node_kind,
            children,
            call,
            op_id: node
                .get("op_id")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
            description: node
                .get("description")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
        });
    }
    Some(Plan {
        plan_id,
        session_id,
        round: 0,
        nodes,
        root_index,
    })
}

/// Memory-service MCP tools (`ptdl_retrieve`, `ptdl_remember`, …) return their
/// JSON payload inside a `{"data": "…"}` envelope — the wire form of the
/// `std_msgs_mcp.String` they hand back to the executor. Unwrap that envelope
/// (recursively, in case a response is double-wrapped) so callers can read the
/// real keys (`plans`, `ok`, …) directly. A payload with no `data` key is
/// returned as-is, so this also works for the older direct-JSON responses.
fn unwrap_mcp_data(output: &str) -> Option<serde_json::Value> {
    let mut v: serde_json::Value = serde_json::from_str(output).ok()?;
    while let Some(data) = v.get("data").and_then(|d| d.as_str()) {
        match serde_json::from_str::<serde_json::Value>(data) {
            Ok(inner) => v = inner,
            Err(_) => break,
        }
    }
    Some(v)
}

// ── PTDL prefetch ──────────────────────────────────────────────────────────

/// Search `ptdl_store.json` for plans similar to *query*.  Returns formatted
/// text suitable for injection into the system prompt, or `None` when the
/// store is empty, the provider is absent, or an error occurs.
///
/// **Explicit opt-in**: planner.rs gates this behind
/// `ROBONIX_MEMORY_PREFETCH_ENABLED=1`.  When disabled (default), planning
/// runs without historical plan context.
pub async fn prefetch(
    query: &str,
    executor: &mut ExecutorConn,
    target: Option<(String, String)>,
) -> Option<String> {
    let (provider_id, contract_id) = target?;

    let payload = serde_json::json!({
        "query": query,
        "top_k": 5,
    });
    let payload_str = serde_json::to_string(&payload).unwrap_or_default();

    let plan = single_call_plan(
        Uuid::new_v4().to_string(),
        "memory-prefetch".to_string(),
        0,
        CapabilityCall {
            call_id: Uuid::new_v4().to_string(),
            provider_id,
            contract_id,
            args_json: serde_json::json!({ "data": payload_str }).to_string(),
        },
        "memory_prefetch",
    );

    let submitted_plan = plan.clone();
    let mut stream = executor
        .graph
        .execute(Request::new(plan))
        .await
        .ok()?
        .into_inner();
    while let Ok(Some(event)) = stream.message().await {
        if event.event_kind == RtdlEventEnum::NodeState as u32
            && let Some(ns) = event.node_state
            && is_terminal_executor_state(ns.state)
        {
            let r = executor_node_state_to_result(&submitted_plan, ns);
            if !r.success || r.output.is_empty() {
                return None;
            }
            // Parse ptdl_retrieve response: {"plans": [...]} (possibly wrapped
            // in the memory service's `{"data": "…"}` MCP envelope).
            let parsed = unwrap_mcp_data(&r.output)?;
            let plans = parsed.get("plans")?.as_array()?;
            if plans.is_empty() {
                return None;
            }
            // Format plans for VLM consumption
            let mut out = String::from("## Similar successful plans\n\n");
            for (i, p) in plans.iter().enumerate() {
                let q = p.get("query").and_then(|v| v.as_str()).unwrap_or("?");
                let d = p.get("description").and_then(|v| v.as_str()).unwrap_or("");
                out.push_str(&format!("**Plan {}:** {}\n", i + 1, q));
                if !d.is_empty() {
                    out.push_str(&format!("  description: {}\n", d));
                }
                // Tree shape: one root record → several sub-plans, each
                // carrying its own steps. Fall back to a flat `steps` array
                // for legacy records.
                if let Some(subs) = p.get("plans").and_then(|v| v.as_array()) {
                    for sub in subs {
                        let sd = sub
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if !sd.is_empty() {
                            out.push_str(&format!("  - {}\n", sd));
                        }
                        if let Some(steps) = sub.get("steps").and_then(|v| v.as_array()) {
                            for s in steps.iter().filter_map(|s| s.as_str()) {
                                out.push_str(&format!("      {}\n", s));
                            }
                        }
                    }
                } else if let Some(steps) = p.get("steps").and_then(|v| v.as_array()) {
                    for s in steps.iter().filter_map(|s| s.as_str()) {
                        out.push_str(&format!("    {}\n", s));
                    }
                }
                out.push('\n');
            }
            debug!("[pilot] memory prefetch: {} plans", plans.len());
            return Some(out);
        }
    }
    None
}

#[derive(Debug)]
pub(crate) struct ReplayPlan {
    pub(crate) plan: Plan,
    pub(crate) saved_plan_id: String,
    pub(crate) expected_success: bool,
}

#[derive(Debug)]
struct SavedPlanAst {
    json: String,
    saved_plan_id: String,
    expected_success: bool,
    stored_index: usize,
}

/// Collect every stored RTDL tree and restore dispatch order from its original
/// numeric plan id. Trees are written when they finish, so JSON insertion order
/// is completion order and is incorrect whenever several trees overlapped.
fn collect_saved_plans(top: &serde_json::Value) -> Vec<SavedPlanAst> {
    let mut plans: Vec<SavedPlanAst> = top
        .get("plans")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(stored_index, sub)| {
            let json = sub.get("rtdl_plan")?.as_str()?.trim();
            if json.is_empty() {
                return None;
            }
            Some(SavedPlanAst {
                json: json.to_string(),
                saved_plan_id: sub
                    .get("plan_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("legacy")
                    .to_string(),
                // Unknown/legacy statuses fail closed: only an explicitly
                // failed or canceled historical tree may fail again without
                // stopping later dependent work.
                expected_success: !matches!(
                    sub.get("status").and_then(|v| v.as_str()),
                    Some("failed" | "canceled")
                ),
                stored_index,
            })
        })
        .collect();
    plans.sort_by(|left, right| {
        match (
            left.saved_plan_id.parse::<u64>(),
            right.saved_plan_id.parse::<u64>(),
        ) {
            (Ok(left_id), Ok(right_id)) => left_id.cmp(&right_id),
            (Ok(_), Err(_)) => std::cmp::Ordering::Less,
            (Err(_), Ok(_)) => std::cmp::Ordering::Greater,
            (Err(_), Err(_)) => left.stored_index.cmp(&right.stored_index),
        }
    });
    plans
}

/// Attempt a direct replay of previously-saved plans for an *identical*
/// question. Every child carrying an RTDL tree is reconstructed in original
/// dispatch order, including failed/canceled children whose side effects or
/// outputs were prerequisites for later recovery rounds. The caller compares
/// each new outcome with the historical status so an expected historical
/// failure can replay through, while a newly-failed successful tree still
/// stops dependent work.
///
/// Returns an empty vec when the store is empty, the question is new, no child
/// carries an RTDL tree, or replay is disabled; normal planning then resumes.
///
/// Gated by `ROBONIX_MEMORY_REPLAY_ENABLED` (default **on**); set to `0` to
/// force re-planning even when an exact match exists.
pub async fn try_replay(
    query: &str,
    executor: &mut ExecutorConn,
    target: Option<(String, String)>,
) -> Vec<ReplayPlan> {
    if std::env::var("ROBONIX_MEMORY_REPLAY_ENABLED")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return Vec::new();
    }
    let Some((provider_id, contract_id)) = target else {
        return Vec::new();
    };

    let payload = serde_json::json!({
        "query": query,
        "top_k": 1,
    });
    let payload_str = serde_json::to_string(&payload).unwrap_or_default();

    let plan = single_call_plan(
        Uuid::new_v4().to_string(),
        "memory-replay".to_string(),
        0,
        CapabilityCall {
            call_id: Uuid::new_v4().to_string(),
            provider_id,
            contract_id,
            args_json: serde_json::json!({ "data": payload_str }).to_string(),
        },
        "memory_replay",
    );

    let submitted_plan = plan.clone();
    let mut stream = match executor.graph.execute(Request::new(plan)).await {
        Ok(resp) => resp.into_inner(),
        Err(_) => return Vec::new(),
    };
    while let Ok(Some(event)) = stream.message().await {
        if event.event_kind == RtdlEventEnum::NodeState as u32
            && let Some(ns) = event.node_state
            && is_terminal_executor_state(ns.state)
        {
            let r = executor_node_state_to_result(&submitted_plan, ns);
            if !r.success || r.output.is_empty() {
                return Vec::new();
            }
            let Some(parsed) = unwrap_mcp_data(&r.output) else {
                return Vec::new();
            };
            let Some(top) = parsed
                .get("plans")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
            else {
                return Vec::new();
            };
            // Exact-match guard: only an identical question replays directly.
            if top
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                != query.trim()
            {
                let _ = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("/tmp/pilot_ptdl_debug.log")
                    .and_then(|mut f| {
                        use std::io::Write;
                        writeln!(
                            f,
                            "REPLAY MISS (not exact): asked={:?} top={:?}",
                            query.trim(),
                            top.get("query").and_then(|v| v.as_str()).unwrap_or("")
                        )
                    });
                return Vec::new();
            }

            let mut saved_plans = collect_saved_plans(top);
            // Legacy flat record fallback (pre-tree shape).
            if saved_plans.is_empty()
                && let Some(ast) = top.get("rtdl_plan").and_then(|v| v.as_str())
                && !ast.trim().is_empty()
            {
                saved_plans.push(SavedPlanAst {
                    json: ast.to_string(),
                    saved_plan_id: "legacy".to_string(),
                    expected_success: true,
                    stored_index: 0,
                });
            }
            if saved_plans.is_empty() {
                let _ = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("/tmp/pilot_ptdl_debug.log")
                    .and_then(|mut f| {
                        use std::io::Write;
                        writeln!(f, "REPLAY MISS (no rtdl_plan): \"{}\"", query.trim())
                    });
                return Vec::new();
            }

            let mut replayed = Vec::with_capacity(saved_plans.len());
            for saved in saved_plans {
                match json_to_plan(
                    &saved.json,
                    Uuid::new_v4().to_string(),
                    "memory-replay".to_string(),
                ) {
                    Some(plan) => replayed.push(ReplayPlan {
                        plan,
                        saved_plan_id: saved.saved_plan_id,
                        expected_success: saved.expected_success,
                    }),
                    None => {
                        let _ = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open("/tmp/pilot_ptdl_debug.log")
                            .and_then(|mut f| {
                                use std::io::Write;
                                writeln!(f, "REPLAY MISS (json_to_plan failed): {}", saved.json)
                            });
                        return Vec::new();
                    }
                }
            }
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/tmp/pilot_ptdl_debug.log")
                .and_then(|mut f| {
                    use std::io::Write;
                    writeln!(
                        f,
                        "REPLAY HIT: \"{query}\" → {} sub-plan(s), {} nodes total",
                        replayed.len(),
                        replayed.iter().map(|p| p.plan.nodes.len()).sum::<usize>()
                    )
                });
            info!(
                "[pilot] memory replay: exact match \"{query}\" → direct dispatch ({} sub-plan(s))",
                replayed.len()
            );
            return replayed;
        }
    }
    Vec::new()
}

// ── PTDL save ──────────────────────────────────────────────────────────────

/// Fire-and-forget: save a successful RTDL plan to `ptdl_store.json` via
/// the `ptdl_remember` MCP tool on the memory service.
///
/// Called from the forest supervisor after a PlanDone with `!any_failed`
/// and `!canceled`.  The caller has already discovered the
/// `robonix/service/memory/ptdl_remember` capability and passes the
/// `(provider_id, contract_id)` target.  Spawns a background task —
/// errors are logged but never propagated (plan-saving is not load-bearing).
#[allow(clippy::too_many_arguments)]
pub fn save_plan(
    executor_graph: RobonixSystemExecutorExecuteClient<tonic::transport::Channel>,
    ptdl_target: (String, String),
    user_query: String,
    plan_description: String,
    steps: Vec<TreeStep>,
    rtdl_plan: Option<String>,
    raw_rtdl: Option<String>,
    tree_count: usize,
    canceled_count: usize,
    plan_id: Option<String>,
) {
    tokio::spawn(async move {
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/pilot_ptdl_debug.log")
            .and_then(|mut f| {
                use std::io::Write;
                writeln!(
                    f,
                    "save_plan ENTERED: \"{q}\" steps={n} trees={t} canceled={c}",
                    q = user_query,
                    n = steps.len(),
                    t = tree_count,
                    c = canceled_count,
                )
            });
        let (provider_id, contract_id) = ptdl_target;

        let steps_text: Vec<String> = steps
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{}. [{}] {}", i + 1, s.capability, s.description))
            .collect();

        let payload = serde_json::json!({
            "query": &user_query,
            "description": &plan_description,
            "steps": &steps_text,
            "plan_count": tree_count,
            "canceled_count": canceled_count,
            "rtdl_plan": rtdl_plan,
            "raw_rtdl": raw_rtdl,
            "plan_id": plan_id,
        });
        let payload_str = serde_json::to_string(&payload).unwrap_or_default();
        let args_json = serde_json::json!({ "data": payload_str }).to_string();

        let plan = single_call_plan(
            Uuid::new_v4().to_string(),
            format!("ptdl-save-{}", Uuid::new_v4()),
            0,
            CapabilityCall {
                call_id: Uuid::new_v4().to_string(),
                provider_id,
                contract_id,
                args_json,
            },
            "ptdl_save",
        );

        let submitted_plan = plan.clone();
        let mut executor = executor_graph;
        let stream_result = executor.execute(Request::new(plan)).await;
        match stream_result {
            Ok(resp) => {
                let mut stream = resp.into_inner();
                while let Ok(Some(event)) = stream.message().await {
                    if event.event_kind == RtdlEventEnum::NodeState as u32
                        && let Some(ns) = event.node_state
                        && is_terminal_executor_state(ns.state)
                    {
                        let r = executor_node_state_to_result(&submitted_plan, ns);
                        if r.success {
                            let _ = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open("/tmp/pilot_ptdl_debug.log")
                                .and_then(|mut f| {
                                    use std::io::Write;
                                    writeln!(f, "save_plan: OK \"{user_query}\"")
                                });
                            info!(
                                "[pilot] ptdl save: \"{user_query}\" ({n} steps)",
                                n = steps_text.len()
                            );
                        } else {
                            let _ = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open("/tmp/pilot_ptdl_debug.log")
                                .and_then(|mut f| {
                                    use std::io::Write;
                                    writeln!(
                                        f,
                                        "save_plan: MEMORY ERROR \"{user_query}\": {}",
                                        r.error
                                    )
                                });
                            warn!(
                                "[pilot] ptdl save: \"{user_query}\" memory error: {}",
                                r.error
                            );
                        }
                        return;
                    }
                }
                let _ = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("/tmp/pilot_ptdl_debug.log")
                    .and_then(|mut f| {
                        use std::io::Write;
                        writeln!(f, "save_plan: NO TERMINAL STATE \"{user_query}\"")
                    });
            }
            Err(status) => {
                let _ = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("/tmp/pilot_ptdl_debug.log")
                    .and_then(|mut f| {
                        use std::io::Write;
                        writeln!(f, "save_plan: RPC FAILED \"{user_query}\": {status}")
                    });
            }
        }
    });
}

// ── Compact (unchanged) ────────────────────────────────────────────────────

/// Best-effort `compact_memory` on session teardown. Logs failures, never
/// propagates errors (the provider may be absent entirely).
pub async fn try_compact(executor: &mut ExecutorConn, atlas: &mut AtlasClient, _consumer_id: &str) {
    let providers = match crate::discovery::discover(atlas).await {
        Ok(c) => c,
        Err(e) => {
            debug!("[pilot] compact_memory: discovery failed: {e}");
            return;
        }
    };
    let Some((provider_id, cap)) = providers
        .iter()
        .find(|(_, cap)| cap.contract_id == "robonix/service/memory/compact")
    else {
        return;
    };

    let plan = single_call_plan(
        Uuid::new_v4().to_string(),
        "memory-compact".to_string(),
        0,
        CapabilityCall {
            call_id: Uuid::new_v4().to_string(),
            provider_id: provider_id.clone(),
            contract_id: cap.contract_id.clone(),
            args_json: "{}".to_string(),
        },
        "memory_compact",
    );

    let submitted_plan = plan.clone();
    let Ok(mut stream) = executor
        .graph
        .execute(Request::new(plan))
        .await
        .map(|r| r.into_inner())
    else {
        return;
    };
    while let Ok(Some(event)) = stream.message().await {
        if event.event_kind == RtdlEventEnum::NodeState as u32
            && let Some(ns) = event.node_state
            && is_terminal_executor_state(ns.state)
        {
            let r = executor_node_state_to_result(&submitted_plan, ns);
            if r.success {
                debug!("[pilot] compact_memory: {}", r.output);
            } else {
                debug!("[pilot] compact_memory failed: {}", r.output);
            }
            return;
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn is_terminal_executor_state(state: u32) -> bool {
    matches!(
        RtdlNodeStateEnum::try_from(state as i32),
        Ok(RtdlNodeStateEnum::Succeeded
            | RtdlNodeStateEnum::Failed
            | RtdlNodeStateEnum::Canceled
            | RtdlNodeStateEnum::Timeout)
    )
}

fn executor_node_state_to_result(
    plan: &Plan,
    ns: crate::pb::pilot::RtdlNodeState,
) -> CapabilityCallResult {
    if let Some(result) = ns.leaf_result {
        return result;
    }
    let call = plan
        .nodes
        .get(ns.node_index as usize)
        .and_then(|node| node.call.as_ref());
    let success = ns.state == RtdlNodeStateEnum::Succeeded as u32;
    CapabilityCallResult {
        call_id: call.map(|c| c.call_id.clone()).unwrap_or_default(),
        provider_id: call.map(|c| c.provider_id.clone()).unwrap_or_default(),
        contract_id: call.map(|c| c.contract_id.clone()).unwrap_or_default(),
        success,
        output: ns.operator_detail.clone(),
        error: if success {
            String::new()
        } else {
            ns.operator_detail
        },
    }
}

#[cfg(test)]
mod tests {
    use super::collect_saved_plans;
    use serde_json::json;

    /// Replay includes failed trees and orders numeric historical ids correctly.
    #[test]
    fn replay_collection_includes_failures_and_restores_plan_id_order() {
        let root = json!({
            "plans": [
                {"plan_id":"3", "status":"success", "rtdl_plan":"three"},
                {"plan_id":"2", "status":"failed", "rtdl_plan":"two"},
                {"plan_id":"21", "status":"success", "rtdl_plan":"twenty-one"},
                {"plan_id":"legacy", "status":"canceled", "rtdl_plan":"legacy"},
                {"plan_id":"unknown", "rtdl_plan":"unknown"}
            ]
        });

        let plans = collect_saved_plans(&root);

        assert_eq!(
            plans
                .iter()
                .map(|plan| plan.saved_plan_id.as_str())
                .collect::<Vec<_>>(),
            vec!["2", "3", "21", "legacy", "unknown"]
        );
        assert!(!plans[0].expected_success);
        assert!(plans[1].expected_success);
        assert!(!plans[3].expected_success);
        assert!(plans[4].expected_success);
    }
}
