#[macro_use] extern crate rocket;

use rocket::State;
use rocket::form::Form;
use rocket::serde::{json::Json, Deserialize, Serialize};
use rocket::fs::{FileServer, TempFile, relative};
use rocket::response::content::RawHtml;
use scraper::{Html, Selector};
use uuid::Uuid;
use serde_json::json;
use fastembed::{
    TextEmbedding, InitOptions, EmbeddingModel, 
    TextRerank, RerankInitOptions, RerankerModel
};
use std::fs;
use std::sync::Arc;
use rocket::response::stream::TextStream;
use futures_util::stream::StreamExt;

// --- Configuration Constants ---
const QDRANT_URL: &str = "http://qdrant:6333";
const COLLECTION_NAME: &str = "aether_chunks";
const VECTOR_SIZE: usize = 384;
const OLLAMA_URL: &str = "http://ollama:11434/api/generate";

// --- Application State ---
struct AppState {
    model: Arc<TextEmbedding>,
    reranker: Arc<TextRerank>,
}

// --- Data Structures ---

#[derive(Deserialize)] 
struct IngestRequest { url: String }

#[derive(Serialize)] 
struct IngestResponse { status: String, message: String }

#[derive(Deserialize)] 
struct AskRequest { question: String }

// Request structure for the evaluation endpoint
#[derive(Deserialize)]
struct EvaluateRequest { question: String, answer: String }

#[derive(Debug, Serialize, Deserialize)]
struct Evaluation {
    faithfulness: f32,
    relevance: f32,
    critique: String,
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

// --- Logic: Core Processing ---

/// Calls LLM to evaluate the generated answer against the context.
async fn run_evaluation_logic(context: &str, question: &str, answer: &str) -> Option<Evaluation> {
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

/// Embeds text chunks and pushes them to Qdrant.
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

/// Splits text into overlapping chunks.
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

/// Serve the frontend index file manually to ensure it loads at root.
#[get("/")]
async fn index() -> Option<RawHtml<String>> {
    fs::read_to_string(relative!("static/index.html")).ok().map(RawHtml)
}

/// Endpoint to evaluate an answer after it has been generated.
#[post("/evaluate", format = "json", data = "<req>")]
async fn evaluate_answer(state: &State<AppState>, req: Json<EvaluateRequest>) -> Option<Json<Evaluation>> {
    let client = reqwest::Client::new();
    let model = state.model.clone();
    
    // 1. Quick retrieval to get the context used for judgment
    let query_vec = model.embed(vec![req.question.clone()], None).ok()?;
    
    let search_res = client.post(format!("{}/collections/{}/points/search", QDRANT_URL, COLLECTION_NAME))
        .json(&json!({ "vector": query_vec[0], "limit": 3, "with_payload": true }))
        .send().await.ok()?;

    let hits = search_res.json::<serde_json::Value>().await.ok()
        .and_then(|v| v["result"].as_array().cloned()).unwrap_or_default();

    let context = hits.iter()
        .map(|h| h["payload"]["text"].as_str().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n---\n");

    // 2. Perform evaluation using the LLM
    let eval = run_evaluation_logic(&context, &req.question, &req.answer).await?;
    Some(Json(eval))
}

/// Main RAG Endpoint: Retrieval -> Reranking -> Streaming Generation.
#[post("/ask", format = "json", data = "<request>")]
async fn ask(state: &State<AppState>, request: Json<AskRequest>) -> TextStream![String] {
    let client = reqwest::Client::new();
    let model = state.model.clone();
    let reranker = state.reranker.clone();
    let question = request.question.clone();

    TextStream! {
        // Step 1: Vector Search (Retrieval)
        let query_vec = match model.embed(vec![question.clone()], None) {
            Ok(v) => v,
            Err(_) => { yield "Error: Embedding failure".to_string(); return; }
        };

        let search_res = client.post(format!("{}/collections/{}/points/search", QDRANT_URL, COLLECTION_NAME))
            .json(&json!({ "vector": query_vec[0], "limit": 10, "with_payload": true }))
            .send().await;

        let hits = match search_res {
            Ok(res) => res.json::<serde_json::Value>().await.ok().and_then(|v| v["result"].as_array().cloned()).unwrap_or_default(),
            Err(_) => vec![],
        };

        if hits.is_empty() {
            yield "No relevant information found in documents.".to_string();
            return;
        }

        // Step 2: Reranking (Cross-Encoder)
        let documents: Vec<String> = hits.iter()
            .map(|h| h["payload"]["text"].as_str().unwrap_or("").to_string())
            .collect();

        let docs_refs: Vec<&String> = documents.iter().collect();

        let rerank_results = match reranker.rerank(&question, docs_refs, false, None) {
            Ok(r) => r,
            Err(_) => { yield "Error: Reranking failure".to_string(); return; }
        };

        let mut ranked_hits: Vec<_> = hits.into_iter().zip(rerank_results.into_iter()).collect();
        ranked_hits.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap());

        let context = ranked_hits.iter()
            .take(3)
            .map(|(h, _)| h["payload"]["text"].as_str().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n---\n");

        // Step 3: Generation (Streaming)
        // Note: System prompt is hardcoded to Italian per user preference, but can be changed.
        let prompt = format!("SYSTEM: Rispondi in italiano usando il contesto.\nCONTESTO:\n{}\nDOMANDA:\n{}", context, question);
        
        let response_res = client.post(OLLAMA_URL)
            .json(&json!({ "model": "llama3.2:3b", "prompt": prompt, "stream": true }))
            .send().await;

        if let Ok(res) = response_res {
            let mut response_stream = res.bytes_stream();
            while let Some(item) = response_stream.next().await {
                if let Ok(bytes) = item {
                    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                        if let Some(token) = json["response"].as_str() { yield token.to_string(); }
                        if json["done"].as_bool().unwrap_or(false) { break; }
                    }
                }
            }
        }
    }
}

/// System Stats Endpoint (CPU/GPU)
#[get("/stats")]
async fn get_stats() -> Json<SystemStats> {
    let cpu_output = std::process::Command::new("sh").arg("-c").arg("top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}'").output().ok();
    let cpu_usage = cpu_output.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_else(|| "0".into());
    let gpu_output = std::process::Command::new("nvidia-smi").arg("--query-gpu=memory.used,memory.total").arg("--format=csv,noheader,nounits").output().ok();
    let (vram_used, vram_total) = if let Some(o) = gpu_output {
        let b = String::from_utf8_lossy(&o.stdout);
        let p: Vec<&str> = b.trim().split(',').collect();
        if p.len() == 2 { (p[0].trim().to_string(), p[1].trim().to_string()) } else { ("0".into(), "0".into()) }
    } else { ("N/A".into(), "N/A".into()) };
    Json(SystemStats { cpu_usage, vram_used, vram_total })
}

/// Resets the Vector DB
#[post("/reset")]
async fn reset() -> Json<IngestResponse> {
    let _ = reqwest::Client::new().delete(format!("{}/collections/{}", QDRANT_URL, COLLECTION_NAME)).send().await;
    init_qdrant().await;
    Json(IngestResponse { status: "success".into(), message: "Database reset complete.".into() })
}

/// Web Scraping Ingestion
#[post("/ingest", format = "json", data = "<task>")]
async fn ingest(state: &State<AppState>, task: Json<IngestRequest>) -> Json<IngestResponse> {
    let client = reqwest::Client::builder().user_agent("AetherEngine/1.0").build().unwrap();
    let response = match client.get(&task.url).send().await {
        Ok(res) => res.text().await.unwrap_or_default(),
        Err(_) => return Json(IngestResponse { status: "error".into(), message: "Web request failed".into() }),
    };

    let text = {
        let document = Html::parse_document(&response);
        let selectors = vec!["#mw-content-text p", "article p", "main p", ".content p"];
        let mut full_text = String::new();
        for sel in selectors {
            if let Ok(s) = Selector::parse(sel) {
                let section: String = document.select(&s).map(|e| e.text().collect::<String>()).collect::<Vec<_>>().join("\n");
                if !section.is_empty() { full_text.push_str(&section); full_text.push('\n'); }
            }
        }
        if full_text.is_empty() {
            Selector::parse("body").ok().map(|s| document.select(&s).map(|e| e.text().collect::<String>()).collect::<String>()).unwrap_or_default()
        } else { full_text }
    };

    match process_and_store(&state.model, text, task.url.clone(), None).await {
        Ok(n) => Json(IngestResponse { status: "success".into(), message: format!("Ingested {} chunks", n) }),
        Err(e) => Json(IngestResponse { status: "error".into(), message: e.to_string() }),
    }
}

/// PDF Upload Ingestion
#[post("/ingest-pdf", data = "<data>")]
async fn ingest_pdf(state: &State<AppState>, mut data: Form<Upload<'_>>) -> Json<IngestResponse> {
    let path = format!("{}.pdf", Uuid::new_v4());
    if data.file.persist_to(&path).await.is_err() { return Json(IngestResponse { status: "error".into(), message: "Filesystem error".into() }); }
    let mut total = 0;
    if let Ok(doc) = lopdf::Document::load(&path) {
        for (p, _) in doc.get_pages() {
            if let Ok(t) = doc.extract_text(&[p]) {
                if let Ok(n) = process_and_store(&state.model, t, "PDF".into(), Some(p)).await { total += n; }
            }
        }
    }
    let _ = fs::remove_file(&path);
    Json(IngestResponse { status: "success".into(), message: format!("Indexed {} chunks", total) })
}

/// Initializes Qdrant collection
async fn init_qdrant() {
    let _ = reqwest::Client::new().put(format!("{}/collections/{}", QDRANT_URL, COLLECTION_NAME))
        .json(&json!({ "vectors": { "size": VECTOR_SIZE, "distance": "Cosine" } })).send().await;
}

#[launch]
async fn rocket() -> _ {
    init_qdrant().await;
    
    // Model Initialization (Mutable to bypass non_exhaustive struct error)
    let mut embed_opts = InitOptions::default();
    embed_opts.model_name = EmbeddingModel::ParaphraseMLMiniLML12V2;
    let model = TextEmbedding::try_new(embed_opts).expect("Failed to load Embedding model");
    
    let mut rerank_opts = RerankInitOptions::default();
    rerank_opts.model_name = RerankerModel::BGERerankerBase;
    let reranker = TextRerank::try_new(rerank_opts).expect("Failed to load Reranker model");

    rocket::build()
        .manage(AppState { model: Arc::new(model), reranker: Arc::new(reranker) })
        .mount("/", FileServer::from(relative!("static")))
        .mount("/", routes![index, evaluate_answer, ingest, ingest_pdf, ask, reset, get_stats])
}