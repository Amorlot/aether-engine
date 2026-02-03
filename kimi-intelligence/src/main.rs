use dotenvy::dotenv;
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};
use std::env;
use std::io::{stdout, Write};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let args: Vec<String> = env::args().collect();
    let user_prompt = args.get(1).map(|s| s.as_str()).unwrap_or("Explain Blackwell FP4 efficiency.");

    let api_key = env::var("NVIDIA_API_KEY")?.trim().to_string();
    
    // Using Microsoft Phi-4 Mini Flash Reasoning from your available models list
    let model_name = "microsoft/phi-4-mini-flash-reasoning"; 
    let endpoint = "https://integrate.api.nvidia.com/v1/chat/completions";

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120)) // Faster model, shorter timeout is fine
        .build()?;

    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key))?);
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let body = json!({
        "model": model_name,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.4,
        "max_tokens": 2048,
        "stream": true
    });

    println!("[STATUS] Requesting {}...", model_name);

    let response = client.post(endpoint)
        .headers(headers)
        .json(&body)
        .send()
        .await?;

    let status = response.status();

    if !status.is_success() {
        let err_body = response.text().await?;
        eprintln!("[ERROR] HTTP {}: {}", status, err_body);
        return Ok(());
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim().to_string();
            buffer = buffer[pos + 1..].to_string();

            if line.starts_with("data: ") {
                let json_str = &line[6..];
                if json_str == "[DONE]" { 
                    println!("\n[STATUS] Stream finished.");
                    return Ok(()); 
                }

                if let Ok(v) = serde_json::from_str::<Value>(json_str) {
                    if let Some(delta) = v["choices"][0].get("delta") {
                        // Phi-4 might use 'reasoning_content' or just 'content' 
                        // depending on the NIM implementation. Checking both:
                        if let Some(reasoning) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                            print!("{}", reasoning);
                        }
                        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
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