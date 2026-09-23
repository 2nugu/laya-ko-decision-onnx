//! Laya decision model over ONNX Runtime.
//!
//! Port of `laya.common.build_sequence` + `rl_agent_api.system_one` post-processing.
//!
//! NOTE: this file has not been compiled — the machine it was written on has no Rust
//! toolchain. The `ort` 2.0 release-candidate API is still in flux, so the session/tensor
//! call sites may need adjusting for the rc you pin. `build_sequence` and the temperature
//! handling are line-by-line ports of the Python and can be checked against
//! `golden.json` independently of the ONNX plumbing.
//! The graph exported with `--logits-only` takes
//!   input_ids[B,L] i64, attention_mask[B,L] i64, marker_pos[B,K] i64,
//!   marker_mask[B,K] bool, qtype[B] i64
//! and returns logits[B,K] f32, already masked with -1e4 on inactive options.

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

// ----------------------------------------------------------------- runtime sidecar
#[derive(Debug, Deserialize)]
pub struct SpecialTokens {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
}

#[derive(Debug, Deserialize)]
pub struct Runtime {
    pub max_len: usize,
    pub head_max_len: usize,
    pub temperature: [f32; 3],
    pub temperature_by_options: HashMap<String, f32>,
    pub special_tokens: SpecialTokens,
    pub mask_token: String,
    pub option_token_cap: usize,
    pub head_min_tokens: usize,
    pub opt_budget_floor: usize,
    pub opt_per_option_floor: usize,
    pub noul_default_false: String,
    pub noul_default_true: String,
}

// ----------------------------------------------------------------- question types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

/// One typed decision. `criteria` carries the option keys and their (optional) descriptions:
///  * choice -> (key, Some(description)) or (key, None)
///  * score  -> (level text, None), in ascending level order
///  * noul   -> empty, or exactly [("false", d0), ("true", d1)]
pub struct Question {
    pub qtype: QType,
    pub instructions: String,
    pub criteria: Vec<(String, Option<String>)>,
}

impl Question {
    /// Port of `render_options`: option texts in label-index order.
    fn render_options(&self, rt: &Runtime) -> Vec<String> {
        match self.qtype {
            QType::Choice => self
                .criteria
                .iter()
                .map(|(k, v)| match v {
                    Some(d) if !d.is_empty() => format!("{k}: {d}"),
                    _ => k.clone(),
                })
                .collect(),
            QType::Score => self
                .criteria
                .iter()
                .enumerate()
                .map(|(i, (c, _))| format!("level {i}: {c}"))
                .collect(),
            QType::Noul => {
                let get = |key: &str, default: &str| -> String {
                    self.criteria
                        .iter()
                        .find(|(k, _)| k == key)
                        .and_then(|(_, v)| v.clone())
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| default.to_string())
                };
                vec![
                    format!("false: {}", get("false", &rt.noul_default_false)),
                    format!("true: {}", get("true", &rt.noul_default_true)),
                ]
            }
        }
    }
}

// ----------------------------------------------------------------- sequence construction
/// Port of `laya.common.build_sequence`.
///
/// Layout: `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`
/// Returns the token ids and the position of each option's `[MASK]` marker.
pub fn build_sequence(
    tok: &Tokenizer,
    rt: &Runtime,
    state: &str,
    q: &Question,
) -> Result<(Vec<u32>, Vec<i64>)> {
    let strip = |s: &str| s.replace(&rt.mask_token, " ");
    let enc = |s: &str| -> Result<Vec<u32>> {
        Ok(tok
            .encode(s, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    };

    let opts = q.render_options(rt);
    let mut head_ids = enc(&format!(
        "{} question: {}",
        q.qtype.name(),
        strip(&q.instructions)
    ))?;

    // each option starts with its own [MASK] marker; option text is capped at 48 tokens
    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(opts.len());
    for o in &opts {
        let mut v = vec![rt.special_tokens.mask];
        let mut body = enc(&format!(" {}", strip(o)))?;
        body.truncate(rt.option_token_cap);
        v.extend(body);
        opt_ids.push(v);
    }

    let used: usize = opt_ids.iter().map(|o| o.len()).sum();
    let mut opt_budget = rt.head_max_len as isize - used as isize;
    if opt_budget < rt.opt_budget_floor as isize {
        // too many / too long options: shrink every option text evenly
        let per = std::cmp::max(
            rt.opt_per_option_floor,
            rt.head_max_len.saturating_sub(rt.opt_budget_floor) / std::cmp::max(1, opt_ids.len()),
        );
        for o in opt_ids.iter_mut() {
            o.truncate(per);
        }
        let used2: usize = opt_ids.iter().map(|o| o.len()).sum();
        opt_budget = rt.head_max_len as isize - used2 as isize;
    }
    head_ids.truncate(std::cmp::max(rt.head_min_tokens as isize, opt_budget) as usize);

    let mut ids: Vec<u32> = Vec::with_capacity(rt.max_len);
    ids.push(rt.special_tokens.cls);
    ids.extend(&head_ids);
    ids.push(rt.special_tokens.sep);

    let mut markers: Vec<i64> = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len() as i64);
        ids.extend(o);
    }
    ids.push(rt.special_tokens.sep);

    let room = rt.max_len.saturating_sub(ids.len() + 1);
    let mut st = enc(&strip(state))?;
    st.truncate(room);
    ids.extend(st);
    ids.push(rt.special_tokens.sep);
    ids.truncate(rt.max_len);

    let limit = rt.max_len as i64;
    markers.retain(|&m| m < limit);
    if markers.len() != opts.len() {
        return Err(anyhow!(
            "options do not fit in head_max_len={}: {} markers for {} options",
            rt.head_max_len,
            markers.len(),
            opts.len()
        ));
    }
    Ok((ids, markers))
}

// ----------------------------------------------------------------- inference
pub struct Laya {
    session: Session,
    tokenizer: Tokenizer,
    pub rt: Runtime,
}

/// Port of `temp_bucket`.
fn temp_bucket(qtype: QType, k: usize) -> String {
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{}", qtype.name(), size)
}

fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

impl Laya {
    pub fn load(dir: impl AsRef<Path>, onnx_file: &str) -> Result<Self> {
        let dir = dir.as_ref();
        let rt: Runtime = serde_json::from_reader(std::fs::File::open(dir.join("laya_runtime.json"))?)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus_or(4))?
            .commit_from_file(dir.join(onnx_file))?;
        Ok(Self { session, tokenizer, rt })
    }

    /// Answer a batch of (state, question) pairs. Returns one probability vector per pair.
    pub fn predict(&mut self, batch: &[(&str, &Question)]) -> Result<Vec<Vec<f32>>> {
        let mut built = Vec::with_capacity(batch.len());
        for (state, q) in batch {
            built.push(build_sequence(&self.tokenizer, &self.rt, state, q)?);
        }
        let b = built.len();
        let l = built.iter().map(|(ids, _)| ids.len()).max().unwrap_or(1);
        let k = built.iter().map(|(_, m)| m.len()).max().unwrap_or(1);

        let mut ids = Array2::<i64>::from_elem((b, l), self.rt.special_tokens.pad as i64);
        let mut att = Array2::<i64>::zeros((b, l));
        let mut pos = Array2::<i64>::zeros((b, k));
        let mut mask = Array2::<bool>::from_elem((b, k), false);
        let mut qt = Array1::<i64>::zeros(b);

        for (i, ((seq, markers), (_, q))) in built.iter().zip(batch.iter()).enumerate() {
            for (j, &t) in seq.iter().enumerate() {
                ids[[i, j]] = t as i64;
                att[[i, j]] = 1;
            }
            for (j, &m) in markers.iter().enumerate() {
                pos[[i, j]] = m;
                mask[[i, j]] = true;
            }
            qt[i] = q.qtype as i64;
        }

        let outputs = self.session.run(ort::inputs![
            "input_ids"     => Value::from_array(ids)?,
            "attention_mask"=> Value::from_array(att)?,
            "marker_pos"    => Value::from_array(pos)?,
            "marker_mask"   => Value::from_array(mask)?,
            "qtype"         => Value::from_array(qt)?,
        ])?;
        let (shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        let kk = shape[1] as usize;

        let mut out = Vec::with_capacity(b);
        for (i, ((_, markers), (_, q))) in built.iter().zip(batch.iter()).enumerate() {
            let n = markers.len();
            let bucket = temp_bucket(q.qtype, n);
            let t = *self
                .rt
                .temperature_by_options
                .get(&bucket)
                .unwrap_or(&self.rt.temperature[q.qtype as usize]);
            let z: Vec<f32> = (0..n).map(|j| logits[i * kk + j] / t).collect();
            out.push(softmax(&z));
        }
        Ok(out)
    }
}

fn num_cpus_or(default: usize) -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(default)
}
