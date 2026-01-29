#[macro_use] extern crate rocket;

use rocket::State;
use rocket::form::Form;
use rocket::serde::{json::Json, Deserialize, Serialize};
use rocket::fs::{FileServer, TempFile};
use scraper::{Html, Selector};
use uuid::Uuid;
use serde_json::json;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::fs;
use std::sync::Arc;

// --- Configuration Constants ---
const QDRANT_URL: &str = "http://qdrant:6333";
const COLLECTION_NAME: &str = "aether_chunks";
const VECTOR_SIZE: usize = 384;
const OLLAMA_URL: &str = "http://ollama:11434/api/generate";

// --- Application State ---
struct AppState {
    model: Arc<TextEmbedding>,
}

// --- Data Structures ---

#[derive(Deserialize)] 
struct IngestRequest { url: String }

#[derive(Serialize)] 
struct IngestResponse { status: String, message: String }

#[derive(Deserialize)] 
struct AskRequest { question: String }

#[derive(Debug, Serialize, Deserialize)]
struct Evaluation {
    faithfulness: f32,
    relevance: f32,
    critique: String,
}

#[derive(Serialize)] 
struct AskResponse { 
    answer: String, 
    thought: String, 
    confidence_score: f32, 
    sources: Vec<SourceMetadata>,
    evaluation: Option<Evaluation>
}

#[derive(Serialize)]
struct SourceMetadata {
    url: String,
    page: Option<u32>,
}

#[derive(Serialize)]
struct SystemStats {
    cpu_usage: String,
    vram_used: String,
    vram_total: String,
}

#[derive(FromForm)]
pub struct Upload<'r> {
    file: TempFile<'r>,
}

// --- Logic: Core Processing & RAG Pipeline ---

/// Evaluates the generated response using a secondary LLM pass (Judge Model).
async fn run_evaluation(context: &str, question: &str, answer: &str) -> Option<Evaluation> {
    let client = reqwest::Client::new();
    let prompt = format!(
        "### System: You are a strict RAG Validator. Evaluate the Answer based ONLY on Context. Respond ONLY with JSON: {{'faithfulness': 0.0-1.0, 'relevance': 0.0-1.0, 'critique': 'str'}}. Context: {context} Question: {question} Answer: {answer} JSON Output:"
    );

    let payload = json!({
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": false,
        "format": "json"
    });

    if let Ok(res) = client.post(OLLAMA_URL).json(&payload).send().await {
        if let Ok(json_body) = res.json::<serde_json::Value>().await {
            let response_text = json_body["response"].as_str().unwrap_or("{}");
            return serde_json::from_str::<Evaluation>(response_text).ok();
        }
    }
    None
}

/// Chunks text, generates embeddings, and pushes vectors to Qdrant.
async fn process_and_store(model: &TextEmbedding, text: String, source: String, page: Option<u32>) -> Result<usize, Box<dyn std::error::Error>> {
    let cleaned_text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let chunks = chunk_text(&cleaned_text, 500, 100); 
    
    if chunks.is_empty() { return Ok(0); }

    let embeddings = model.embed(chunks.clone(), None)?;
    
    let points: Vec<serde_json::Value> = chunks.iter().enumerate().map(|(i, chunk)| {
        json!({ 
            "id": Uuid::new_v4().to_string(), 
            "vector": embeddings[i], 
            "payload": { 
                "text": chunk, 
                "url": source.clone(),
                "page": page 
            } 
        })
    }).collect();

    reqwest::Client::new()
        .put(format!("{}/collections/{}/points?wait=true", QDRANT_URL, COLLECTION_NAME))
        .json(&json!({ "points": points }))
        .send()
        .await?;
        
    Ok(chunks.len())
}

/// Splits text into fixed-size chunks with overlap for semantic continuity.
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        chunks.push(chars[start..end].iter().collect());
        if end == chars.len() { break; }
        start += chunk_size - overlap;
    }
    chunks
}

// --- API Endpoints ---

/// Fetches real-time system resource metrics (CPU and VRAM).
#[get("/stats")]
async fn get_stats() -> Json<SystemStats> {
    // CPU Monitoring via 'top'
    let cpu_output = std::process::Command::new("sh")
        .arg("-c")
        .arg("top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}'")
        .output().ok();
    let cpu_usage = cpu_output.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "0".into());

    // GPU Monitoring via 'nvidia-smi'
    let gpu_output = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.used,memory.total")
        .arg("--format=csv,noheader,nounits")
        .output().ok();

    let (vram_used, vram_total) = if let Some(o) = gpu_output {
        let binding = String::from_utf8_lossy(&o.stdout);
        let parts: Vec<&str> = binding.trim().split(',').collect();
        if parts.len() == 2 { 
            (parts[0].trim().to_string(), parts[1].trim().to_string()) 
        } else { ("0".into(), "0".into()) }
    } else { ("N/A".into(), "N/A".into()) };

    Json(SystemStats { cpu_usage, vram_used, vram_total })
}

/// Purges current vector collection and re-initializes.
#[post("/reset")]
async fn reset() -> Json<IngestResponse> {
    let client = reqwest::Client::new();
    let _ = client.delete(format!("{}/collections/{}", QDRANT_URL, COLLECTION_NAME)).send().await;
    init_qdrant().await;
    Json(IngestResponse { status: "success".into(), message: "Collection reset successfully.".into() })
}

/// Web scraping ingestion endpoint.
#[post("/ingest", format = "json", data = "<task>")]
async fn ingest(state: &State<AppState>, task: Json<IngestRequest>) -> Json<IngestResponse> {
    let response = match reqwest::get(&task.url).await {
        Ok(res) => res.text().await.unwrap_or_default(),
        Err(_) => return Json(IngestResponse { status: "error".into(), message: "Web request failed".into() }),
    };

    let text = {
        let document = Html::parse_document(&response);
        let wiki_selector = Selector::parse("#mw-content-text p").ok();
        let body_selector = Selector::parse("body").ok();

        let mut extracted = wiki_selector
            .map(|s| document.select(&s).map(|e| e.text().collect::<Vec<_>>().join("")).collect::<Vec<_>>().join("\n"))
            .unwrap_or_default();

        if extracted.is_empty() {
            extracted = body_selector
                .and_then(|s| document.select(&s).next().map(|b| b.text().collect::<Vec<_>>().join(" ")))
                .unwrap_or_default();
        }
        extracted
    };

    match process_and_store(&state.model, text, task.url.clone(), None).await {
        Ok(n) => Json(IngestResponse { status: "success".into(), message: format!("Ingested {} chunks", n) }),
        Err(e) => Json(IngestResponse { status: "error".into(), message: e.to_string() }),
    }
}

/// PDF document ingestion endpoint.
#[post("/ingest-pdf", data = "<data>")]
async fn ingest_pdf(state: &State<AppState>, mut data: Form<Upload<'_>>) -> Json<IngestResponse> {
    let temp_path = format!("{}.pdf", Uuid::new_v4());
    if data.file.persist_to(&temp_path).await.is_err() { 
        return Json(IngestResponse { status: "error".into(), message: "FS persistence failed".into() }); 
    }

    let mut total_chunks = 0;
    let mut page_count = 0;

    if let Ok(doc) = lopdf::Document::load(&temp_path) {
        for (page_num, _) in doc.get_pages() {
            if let Ok(text) = doc.extract_text(&[page_num]) {
                if let Ok(n) = process_and_store(&state.model, text, "Uploaded PDF".into(), Some(page_num)).await {
                    total_chunks += n;
                    page_count += 1;
                }
            }
        }
    } else {
         // Fallback if lopdf fails to load structured pages
         let _ = fs::remove_file(&temp_path);
         return Json(IngestResponse { status: "error".into(), message: "PDF Load Error".into() });
    }

    let _ = fs::remove_file(&temp_path);
    Json(IngestResponse { status: "success".into(), message: format!("Indexed {} chunks across {} pages", total_chunks, page_count) })
}

/// Main Inference endpoint: Semantic Search -> Context Augmentation -> Generation.
#[post("/ask", format = "json", data = "<request>")]
async fn ask(state: &State<AppState>, request: Json<AskRequest>) -> Json<AskResponse> {
    let client = reqwest::Client::new();
    
    // 1. Vector Search via Qdrant
    let query_vec = match state.model.embed(vec![request.question.clone()], None) {
        Ok(v) => v,
        Err(_) => return Json(AskResponse { answer: "Embedding failure".into(), thought: "".into(), confidence_score: 0.0, sources: vec![], evaluation: None }),
    };

    let search_res = client.post(format!("{}/collections/{}/points/search", QDRANT_URL, COLLECTION_NAME))
        .json(&json!({ "vector": query_vec[0], "limit": 8, "with_payload": true }))
        .send().await;

    let hits = match search_res {
        Ok(res) => res.json::<serde_json::Value>().await.ok().and_then(|v| v["result"].as_array().cloned()).unwrap_or_default(),
        Err(_) => vec![],
    };

    let top_score = hits.first().and_then(|h| h["score"].as_f64()).unwrap_or(0.0) as f32;
    let context = hits.iter().map(|h| h["payload"]["text"].as_str().unwrap_or("")).collect::<Vec<_>>().join("\n---\n");
    
    let sources = hits.iter().map(|h| SourceMetadata {
        url: h["payload"]["url"].as_str().unwrap_or("Unknown").to_string(),
        page: h["payload"]["page"].as_u64().map(|p| p as u32),
    }).collect();

    // 2. Generation via Ollama
    let prompt = format!("SYSTEM: Rispondi in italiano usando solo il contesto fornito.\nTHOUGHT: [ragionamento]\nANSWER: [risposta]\nCONTESTO:\n{}\nDOMANDA:\n{}", context, request.question);

    let ollama_res = client.post(OLLAMA_URL)
        .json(&json!({ "model": "llama3.2:3b", "prompt": prompt, "stream": false }))
        .send().await;

    let (thought, answer) = match ollama_res {
        Ok(res) => {
            let body: serde_json::Value = res.json().await.unwrap_or_default();
            let full_response = body["response"].as_str().unwrap_or("").to_string();
            if let Some(pos) = full_response.find("ANSWER:") {
                (full_response[..pos].replace("THOUGHT:", "").trim().to_string(), full_response[pos..].replace("ANSWER:", "").trim().to_string())
            } else { ("Analysis complete.".to_string(), full_response) }
        },
        Err(_) => ("System failure.".into(), "Inference unavailable.".into()),
    };

    // 3. Post-Generation Evaluation
    let evaluation = run_evaluation(&context, &request.question, &answer).await;

    Json(AskResponse { answer, thought, confidence_score: top_score, sources, evaluation })
}

/// Ensures Qdrant collection existence on startup.
async fn init_qdrant() {
    let _ = reqwest::Client::new()
        .put(format!("{}/collections/{}", QDRANT_URL, COLLECTION_NAME))
        .json(&json!({ "vectors": { "size": VECTOR_SIZE, "distance": "Cosine" } }))
        .send()
        .await;
}

#[launch]
async fn rocket() -> _ {
    init_qdrant().await;
    
    // FIX APPLICATO QUI:
    // Creiamo le opzioni mutabili per impostare il modello corretto
    let mut options = InitOptions::default();
    options.model_name = EmbeddingModel::ParaphraseMLMiniLML12V2;

    let model = TextEmbedding::try_new(options).expect("Critical: Embedding model failed to load.");

    let state = AppState { model: Arc::new(model) };
    
    let figment = rocket::Config::figment()
        .merge(("limits.forms", 10 * 1024 * 1024))
        .merge(("address", "0.0.0.0"))
        .merge(("port", 8000));

    rocket::custom(figment)
        .manage(state)
        .mount("/", FileServer::from("static"))
        .mount("/", routes![ingest, ingest_pdf, ask, reset, get_stats])
}