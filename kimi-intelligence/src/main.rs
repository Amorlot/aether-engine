use dotenvy::dotenv;
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};
use std::env;
use std::io::{stdout, Write};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env file
    dotenv().ok();

    // Retrieve and clean API Key
    let api_key = env::var("NVIDIA_API_KEY")
        .expect("NVIDIA_API_KEY not found in .env")
        .trim() // Remove potential hidden spaces or newlines
        .to_string();

    let args: Vec<String> = env::args().collect();
    let user_prompt = args.get(1).map(|s| s.as_str()).unwrap_or("Hello, Kimi!");

    // Use the URL from .env or fallback to default
    let api_url = env::var("NVIDIA_API_URL")
        .unwrap_or_else(|_| "https://integrate.api.nvidia.com/v1".to_string());
    
    let endpoint = format!("{}/chat/completions", api_url.trim_end_matches('/'));

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    let mut headers = HeaderMap::new();
    // Ensure the Bearer token is formatted correctly
    let auth_header = format!("Bearer {}", api_key);
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_header)?);
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let body = json!({
        "model": "moonshotai/kimi-k2.5",
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.3,
        "max_tokens": 1024,
        "stream": true
    });

    println!("[DEBUG] Using endpoint: {}", endpoint);
    println!("[STATUS] Connecting...");

    let response = client.post(&endpoint)
        .headers(headers)
        .json(&body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let err_text = response.text().await?;
        eprintln!("[ERROR] Authentication failed or API error.");
        eprintln!("[DETAILS] Status: {} - Body: {}", status, err_text);
        return Ok(());
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk_result) = stream.next().await {
        let bytes = chunk_result?;
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim().to_string();
            buffer = buffer[pos + 1..].to_string();

            if line.starts_with("data: ") {
                let json_str = &line[6..];
                if json_str == "[DONE]" { 
                    println!("\n[STATUS] Finished.");
                    return Ok(()); 
                }

                if let Ok(v) = serde_json::from_str::<Value>(json_str) {
                    if let Some(delta) = v["choices"][0].get("delta") {
                        if let Some(reasoning) = delta["reasoning_content"].as_str() {
                            print!("{}", reasoning);
                        }
                        if let Some(content) = delta["content"].as_str() {
                            print!("{}", content);
                        }
                        stdout().flush()?;
                    }
                }
            }
        }
    }

    Ok(())
}