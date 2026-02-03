use dotenvy::dotenv;
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};
use std::env;
use std::io::{stdout, Write};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenv().ok();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run -- \"Prompt\"");
        std::process::exit(1);
    }
    let user_prompt = &args[1];

    let api_key = env::var("NVIDIA_API_KEY").expect("NVIDIA_API_KEY not found");
    let api_url = env::var("NVIDIA_API_URL").unwrap_or_else(|_| "https://integrate.api.nvidia.com/v1".to_string());

    // Configure client with timeout to prevent infinite hangs
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key))?);
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let body = json!({
        "model": "moonshotai/kimi-k2.5",
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.3,
        "stream": true
    });

    println!("[STATUS] Sending request to NVIDIA NIM...");
    stdout().flush()?;

    let response = client.post(format!("{}/chat/completions", api_url))
        .headers(headers)
        .json(&body)
        .send()
        .await?;

    let status = response.status();
    println!("[STATUS] HTTP Response received: {}", status);

    if !status.is_success() {
        let err_text = response.text().await?;
        eprintln!("[ERROR] Error details: {}", err_text);
        return Ok(());
    }

    println!("[STATUS] Waiting for tokens (Reasoning phase)...");
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        let text = String::from_utf8_lossy(&bytes);

        for line in text.lines() {
            if line.starts_with("data: ") {
                let json_str = &line[6..];
                if json_str == "[DONE]" { break; }

                if let Ok(v) = serde_json::from_str::<Value>(json_str) {
                    let delta = &v["choices"][0]["delta"];

                    // Display reasoning content (Thinking process)
                    if let Some(reasoning) = delta["reasoning_content"].as_str() {
                        print!("{}", reasoning);
                        stdout().flush()?;
                    }

                    // Display final response content
                    if let Some(content) = delta["content"].as_str() {
                        print!("{}", content);
                        stdout().flush()?;
                    }
                }
            }
        }
    }

    println!("\n[STATUS] Stream completed.");
    Ok(())
}