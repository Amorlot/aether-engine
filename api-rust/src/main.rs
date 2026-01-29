#[macro_use] extern crate rocket;

use rocket::serde::{json::Json, Deserialize, Serialize};

// Data structure for incoming requests
// The 'Deserialize' trait allows converting JSON input into this Rust struct
#[derive(Deserialize)]
struct IngestRequest {
    url: String,
}

// Data structure for API responses
// The 'Serialize' trait allows converting this Rust struct back into JSON format
#[derive(Serialize)]
struct IngestResponse {
    status: String,
    message: String,
}

// Handler for the POST /ingest endpoint
// It receives a JSON body mapped to the IngestRequest struct
#[post("/ingest", format = "json", data = "<task>")]
async fn ingest(task: Json<IngestRequest>) -> Json<IngestResponse> {
    
    // Log the incoming URL to the console for debugging
    println!("Received ingestion request for: {}", task.url);
    
    // BUSINESS LOGIC PLACEHOLDER: 
    // This is where we will eventually trigger the web scraper and the AI worker
    
    // Return a JSON response confirming the task was accepted
    Json(IngestResponse {
        status: "accepted".to_string(),
        message: format!("Processing started for URL: {}", task.url),
    })
}

// The 'launch' macro generates the main function and starts the server
#[launch]
fn rocket() -> _ {
    // Build the Rocket instance and mount the routes at the root path ("/")
    rocket::build().mount("/", routes![ingest])
}