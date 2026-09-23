//! Minimal end-to-end example.
//!
//!   cargo run --release --example classify -- /path/to/onnx-dir
//!
//! The directory must contain `model.onnx`, `model.onnx.data`, `tokenizer.json`
//! and `laya_runtime.json` (download them from the Hugging Face repo).

use anyhow::Result;
use laya_ko::{Laya, QType, Question};

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "onnx".to_string());
    let mut model = Laya::load(&dir, "model.onnx")?;

    // 1. choice — Korean news topic classification (the KLUE-YNAT label set)
    let topic = Question {
        qtype: QType::Choice,
        instructions: "다음 뉴스 제목의 주제 분야를 고르세요.".into(),
        criteria: vec![
            ("IT과학".into(), Some("정보기술·과학 기사".into())),
            ("경제".into(), Some("경제·금융·산업 기사".into())),
            ("사회".into(), Some("사회·사건사고 기사".into())),
            ("생활문화".into(), Some("생활·문화·연예 기사".into())),
            ("세계".into(), Some("국제·해외 기사".into())),
            ("스포츠".into(), Some("스포츠 기사".into())),
            ("정치".into(), Some("정치·외교 기사".into())),
        ],
    };

    // 2. noul — a yes/no judgement
    let grounded = Question {
        qtype: QType::Noul,
        instructions: "이 문장이 경제 지표를 언급하고 있는가?".into(),
        criteria: vec![],
    };

    // 3. score — an ordinal rating
    let sentiment = Question {
        qtype: QType::Score,
        instructions: "이 문장의 감성을 판정하세요.".into(),
        criteria: vec![
            ("매우 부정적".into(), None),
            ("부정적".into(), None),
            ("중립적".into(), None),
            ("긍정적".into(), None),
            ("매우 긍정적".into(), None),
        ],
    };

    let state = "한국은행, 기준금리 0.25%p 인하 결정";

    // All three go through in one batched forward pass.
    let probs = model.predict(&[(state, &topic), (state, &grounded), (state, &sentiment)])?;

    let labels: Vec<&str> = topic.criteria.iter().map(|(k, _)| k.as_str()).collect();
    let best = argmax(&probs[0]);
    println!("state    : {state}");
    println!("topic    : {} (p={:.3})", labels[best], probs[0][best]);
    println!("grounded : p(true)={:.3}", probs[1][1]);

    // A `score` answer is conventionally read as the mean level, not the argmax.
    let mean: f32 = probs[2].iter().enumerate().map(|(i, p)| i as f32 * p).sum();
    println!("sentiment: level {mean:.2} of 0..4");
    Ok(())
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
