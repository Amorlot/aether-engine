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

// --- Constants ---
const QDRANT_URL: &str = "http://qdrant:6333";
const COLLECTION_NAME: &str = "aether_chunks";
const VECTOR_SIZE: usize = 384;

// --- App State (Shared Model) ---
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

#[derive(Serialize)] 
struct AskResponse { 
    answer: String, 
    thought: String, 
    confidence_score: f32, 
    sources: Vec<String> 
}

#[derive(FromForm)]
pub struct Upload<'r> {
    file: TempFile<'r>,
}

// --- Logic: Processing and Storing ---

async fn process_and_store(model: &TextEmbedding, text: String, source: String) -> Result<usize, Box<dyn std::error::Error>> {
    let cleaned_text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let chunks = chunk_text(&cleaned_text, 500, 100); 
    
    if chunks.is_empty() { return Ok(0); }

    let embeddings = model.embed(chunks.clone(), None)?;

    let points: Vec<serde_json::Value> = chunks.iter().enumerate().map(|(i, chunk)| {
        json!({
            "id": Uuid::new_v4().to_string(),
            "vector": embeddings[i],
            "payload": { "text": chunk, "url": source }
        })
    }).collect();

    let client = reqwest::Client::new();
    client.put(format!("{}/collections/{}/points?wait=true", QDRANT_URL, COLLECTION_NAME))
        .json(&json!({ "points": points }))
        .send().await?;
    
    Ok(chunks.len())
}

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

// --- Endpoints ---

#[post("/reset")]
async fn reset() -> Json<IngestResponse> {
    let client = reqwest::Client::new();
    let _ = client.delete(format!("{}/collections/{}", QDRANT_URL, COLLECTION_NAME)).send().await;
    init_qdrant().await;
    Json(IngestResponse { status: "success".into(), message: "Memory cleared.".into() })
}

#[post("/ingest", format = "json", data = "<task>")]
async fn ingest(state: &State<AppState>, task: Json<IngestRequest>) -> Json<IngestResponse> {
    let response = match reqwest::get(&task.url).await {
        Ok(res) => res.text().await.unwrap_or_default(),
        Err(_) => return Json(IngestResponse { status: "error".into(), message: "Web request failed".into() }),
    };
    
    let text = {
        let document = Html::parse_document(&response);
        let wiki_selector = Selector::parse("#mw-content-text p").unwrap();
        let body_selector = Selector::parse("body").unwrap();
        let mut extracted = document.select(&wiki_selector).map(|e| e.text().collect::<Vec<_>>().join("")).collect::<Vec<_>>().join("\n");
        if extracted.is_empty() {
            extracted = document.select(&body_selector).next().map(|b| b.text().collect::<Vec<_>>().join(" ")).unwrap_or_default();
        }
        extracted
    };

    match process_and_store(&state.model, text, task.url.clone()).await {
        Ok(n) => Json(IngestResponse { status: "success".into(), message: format!("Stored {} chunks", n) }),
        Err(e) => Json(IngestResponse { status: "error".into(), message: e.to_string() }),
    }
}

#[post("/ingest-pdf", data = "<data>")]
async fn ingest_pdf(state: &State<AppState>, mut data: Form<Upload<'_>>) -> Json<IngestResponse> {
    let temp_path = format!("{}.pdf", Uuid::new_v4());
    
    if data.file.persist_to(&temp_path).await.is_err() {
        return Json(IngestResponse { status: "error".into(), message: "Failed to save PDF".into() });
    }

    let text = match pdf_extract::extract_text(&temp_path) {
        Ok(t) => {
            println!("DEBUG: Extracted {} characters from PDF", t.len());
            t
        },
        Err(e) => {
            let _ = fs::remove_file(&temp_path);
            return Json(IngestResponse { status: "error".into(), message: format!("Extraction error: {}", e) });
        }
    };

    let _ = fs::remove_file(&temp_path);
    
    if text.trim().is_empty() {
        return Json(IngestResponse { status: "error".into(), message: "PDF contains no readable text".into() });
    }

    match process_and_store(&state.model, text, "Uploaded PDF".into()).await {
        Ok(n) => Json(IngestResponse { status: "success".into(), message: format!("Stored {} chunks", n) }),
        Err(e) => Json(IngestResponse { status: "error".into(), message: e.to_string() }),
    }
}

#[post("/ask", format = "json", data = "<request>")]
async fn ask(state: &State<AppState>, request: Json<AskRequest>) -> Json<AskResponse> {
    let query_vec = state.model.embed(vec![request.question.clone()], None).unwrap();

    let client = reqwest::Client::new();
    let search_res = client.post(format!("{}/collections/{}/points/search", QDRANT_URL, COLLECTION_NAME))
        .json(&json!({ "vector": query_vec[0], "limit": 8, "with_payload": true }))
        .send().await.unwrap();

    let body: serde_json::Value = search_res.json().await.unwrap();
    let hits = body["result"].as_array().expect("Search failed");
    let top_score = hits.first().and_then(|h| h["score"].as_f64()).unwrap_or(0.0) as f32;

    println!("DEBUG: Best Search Score: {:.4}", top_score);

    if top_score < 0.20 {
        return Json(AskResponse {
            answer: "No relevant information found in current context.".into(),
            thought: format!("Confidence too low ({:.4}). Text might be unreadable or not indexed.", top_score).into(),
            confidence_score: top_score,
            sources: vec![],
        });
    }

    let context = hits.iter()
        .map(|h| h["payload"]["text"].as_str().unwrap_or(""))
        .collect::<Vec<_>>().join("\n---\n");

    let prompt = format!(
        "SYSTEM: Answer using the context. Format your response with THOUGHT: and ANSWER:.\n\
        Context: {}\n\nQuestion: {}\n\n",
        context, request.question
    );

    let ollama_res = client.post("http://ollama:11434/api/generate")
        .json(&json!({ "model": "llama3.2:3b", "prompt": prompt, "stream": false }))
        .send().await.unwrap();

    let ollama_body: serde_json::Value = ollama_res.json().await.unwrap();
    let full_response = ollama_body["response"].as_str().unwrap_or("").to_string();

    let (thought, answer) = if let Some(pos) = full_response.find("ANSWER:") {
        (
            full_response[..pos].replace("THOUGHT:", "").trim().to_string(),
            full_response[pos..].replace("ANSWER:", "").trim().to_string()
        )
    } else { ("Analysis complete.".to_string(), full_response) };

    Json(AskResponse { 
        answer, 
        thought, 
        confidence_score: top_score, 
        sources: hits.iter().map(|h| h["payload"]["url"].as_str().unwrap().to_string()).collect() 
    })
}

async fn init_qdrant() {
    let client = reqwest::Client::new();
    let _ = client.put(format!("{}/collections/{}", QDRANT_URL, COLLECTION_NAME))
        .json(&json!({ "vectors": { "size": VECTOR_SIZE, "distance": "Cosine" } }))
        .send().await;
}

#[launch]
async fn rocket() -> _ {
    init_qdrant().await;

    let mut options = InitOptions::default();
    options.model_name = EmbeddingModel::ParaphraseMLMiniLML12V2;
    let model = TextEmbedding::try_new(options).expect("Model init failed");
    
    let state = AppState {
        model: Arc::new(model),
    };

    // Configuration: Allow 10MB form uploads
    let figment = rocket::Config::figment()
        .merge(("limits.forms", 10 * 1024 * 1024))
        .merge(("address", "0.0.0.0"))
        .merge(("port", 8000));

    rocket::custom(figment)
        .manage(state)
        .mount("/", FileServer::from("static"))
        .mount("/", routes![ingest, ingest_pdf, ask, reset])
}