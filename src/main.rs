// Constants
// Message constants
const CHAT_COMPLETION_OBJECT: &str = "chat.completion.chunk";
const ASSISTANT_ROLE: &str = "assistant";
const SSE_DONE_MESSAGE: &str = "[DONE]";
const STOP_REASON: &str = "stop";

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::{
    Client,
    types::{ContentBlock, ConversationRole, Message, SystemContentBlock},
};
use axum::{
    Json, Router,
    http::StatusCode,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::post,
};
use chrono::prelude::*;
use either::Either;
use futures::stream::Stream;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use thiserror::Error;
use uuid::Uuid;
use void::Void;

mod good;
use good::{
    ChatCompletionsRequest, Content, MessageContent, OpenaiMessage, Role, process_content,
    process_system_content,
};

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Internal server error: {0}")]
    Internal(#[from] anyhow::Error),
    #[error("Stream error: {0}")]
    StreamError(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, error_message) = match self {
            ApiError::Internal(ref e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            ApiError::StreamError(ref e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };

        let body = Json(serde_json::json!({
            "error": {
                "message": error_message,
                "type": "server_error",
                "status": status.as_u16()
            }
        }));

        (status, body).into_response()
    }
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionChunk {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<ChatCompletionChunkChoice>,
    usage: Option<Usage>,
}

pub struct ChatCompletionChunkBuilder {
    chunk: ChatCompletionChunk,
}

impl Default for ChatCompletionChunkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatCompletionChunkBuilder {
    pub fn new() -> Self {
        ChatCompletionChunkBuilder {
            chunk: ChatCompletionChunk {
                id: Uuid::new_v4().to_string(),
                object: CHAT_COMPLETION_OBJECT.to_string(),
                created: Utc::now().timestamp(),
                model: String::new(),
                choices: Vec::new(),
                usage: None,
            },
        }
    }

    pub fn id(mut self, id: String) -> Self {
        self.chunk.id = id;
        self
    }

    pub fn object(mut self, object: String) -> Self {
        self.chunk.object = object;
        self
    }

    pub fn created(mut self, created: i64) -> Self {
        self.chunk.created = created;
        self
    }

    pub fn model(mut self, model: String) -> Self {
        self.chunk.model = model;
        self
    }

    pub fn choices(mut self, choices: Vec<ChatCompletionChunkChoice>) -> Self {
        self.chunk.choices = choices;
        self
    }

    pub fn usage(mut self, usage: Usage) -> Self {
        self.chunk.usage = Some(usage);
        self
    }

    pub fn build(self) -> ChatCompletionChunk {
        self.chunk
    }
}

#[derive(Serialize, Debug)]
pub struct Usage {
    completion_tokens: i32,
    prompt_tokens: i32,
    total_tokens: i32,
    completion_tokens_details: Option<serde_json::Value>,
    prompt_tokens_details: Option<serde_json::Value>,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionChunkChoice {
    delta: ChatCompletionChunkChoiceDelta,
    index: i32,
    finish_reason: Option<String>,
}

#[derive(Serialize, Debug)]
#[serde(untagged)]
pub enum ChatCompletionChunkChoiceDelta {
    Role { role: String },
    Content { content: String },
}

#[async_trait]
pub trait ChatProvider {
    async fn chat_completions_stream(
        self,
        request: ChatCompletionsRequest,
    ) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError>;
}

pub struct BedrockProvider {}

impl BedrockProvider {
    pub async fn new() -> Self {
        BedrockProvider {}
    }
}

#[async_trait]
impl ChatProvider for BedrockProvider {
    async fn chat_completions_stream(
        self,
        payload: ChatCompletionsRequest,
    ) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
        let (system_content_blocks, messages) = process_messages(&payload.messages);

        // Generate consistent id and timestamp for all chunks
        let completion_id = Uuid::new_v4().to_string();
        let created_timestamp = Utc::now().timestamp();

        // Create the client here
        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .region(DEFAULT_AWS_REGION)
            .load()
            .await;
        let client = Client::new(&sdk_config);

        let mut stream = client
            .converse_stream()
            .model_id(&payload.model)
            .set_system(Some(system_content_blocks))
            .set_messages(Some(messages))
            .send()
            .await
            .map_err(|e| ApiError::Internal(e.into()))?
            .stream;

        let sse_stream = async_stream::stream! {
            loop {
                match stream.recv().await {
                    Ok(Some(event)) => {
                        if let sse_event = handle_stream_event(&payload.model, &completion_id, created_timestamp, event) {
                            yield Ok(sse_event);
                        }
                    }
                    Ok(None) => {
                        // Stream completed normally
                        break;
                    }
                    Err(e) => {
                        tracing::error!("Stream error: {}", e);
                        // Send an error finish reason
                        let error_chunk = ChatCompletionChunkBuilder::new()
                            .id(completion_id.clone())
                            .created(created_timestamp)
                            .model(payload.model.clone())
                            .choices(vec![ChatCompletionChunkChoice {
                                delta: ChatCompletionChunkChoiceDelta::Content {
                                    content: String::new(),
                                },
                                index: 0,
                                finish_reason: Some("error".to_string()),
                            }])
                            .build();
                        yield Ok(create_sse_event(&error_chunk));
                        break;
                    }
                }
            }
            // Send final [DONE] message
            yield Ok(Event::default().data(SSE_DONE_MESSAGE));
        };

        Ok(Sse::new(sse_stream))
    }
}

const DEFAULT_AWS_REGION: &str = "us-east-1";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let app = Router::new().route("/chat/completions", post(chat_completions));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn process_conversation_message(msg: &OpenaiMessage) -> Option<Message> {
    let content_blocks = process_content(&msg.content);
    let role = match msg.role {
        Role::Assistant => ConversationRole::Assistant,
        Role::User => ConversationRole::User,
        Role::System => return None, // Should not happen due to partition_map
    };

    Message::builder()
        .role(role)
        .set_content(Some(content_blocks))
        .build()
        .ok()
}

fn process_messages(messages: &[OpenaiMessage]) -> (Vec<SystemContentBlock>, Vec<Message>) {
    let (system_blocks, conversation_messages): (Vec<_>, Vec<_>) =
        messages.iter().partition_map(|msg| match msg.role {
            // System messages go to the left branch
            Role::System => Either::Left(process_system_content(&msg.content)),
            // Conversation messages (User/Assistant) go to the right branch
            Role::Assistant | Role::User => Either::Right(process_conversation_message(msg)),
        });

    (
        // Flatten the nested system blocks into a single vector
        system_blocks.into_iter().flatten().collect(),
        // Filter out None values from conversation messages
        conversation_messages.into_iter().flatten().collect(),
    )
}

/// Handles different stream events and creates appropriate chunks
fn handle_stream_event(
    model: &str,
    completion_id: &str,
    created_timestamp: i64,
    event: aws_sdk_bedrockruntime::types::ConverseStreamOutput,
) -> Event {
    let chunk = match event {
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockDelta(event) => {
            let content = event
                .delta
                .and_then(|d| match d {
                    aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(text) => Some(text),
                    _ => None,
                })
                .unwrap_or_default();

            ChatCompletionChunkBuilder::new()
                .id(completion_id.to_string())
                .created(created_timestamp)
                .id(completion_id.to_string())
                .created(created_timestamp)
                .model(model.to_string())
                .choices(vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Content { content },
                    index: 0,
                    finish_reason: None,
                }])
                .build()
        }
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockStart(_)
        | aws_sdk_bedrockruntime::types::ConverseStreamOutput::MessageStart(_) => {
            ChatCompletionChunkBuilder::new()
                .id(completion_id.to_string())
                .created(created_timestamp)
                .model(model.to_string())
                .choices(vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Role {
                        role: ASSISTANT_ROLE.to_string(),
                    },
                    index: 0,
                    finish_reason: None,
                }])
                .build()
        }
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockStop(_)
        | aws_sdk_bedrockruntime::types::ConverseStreamOutput::MessageStop(_) => {
            ChatCompletionChunkBuilder::new()
                .id(completion_id.to_string())
                .created(created_timestamp)
                .model(model.to_string())
                .choices(vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Content {
                        content: String::new(),
                    },
                    index: 0,
                    finish_reason: Some(STOP_REASON.to_string()),
                }])
                .build()
        }
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::Metadata(event) => {
            if let Some(usage) = event.usage {
                ChatCompletionChunkBuilder::new()
                    .model(model.to_string())
                    .choices(vec![ChatCompletionChunkChoice {
                        delta: ChatCompletionChunkChoiceDelta::Content {
                            content: String::new(),
                        },
                        index: 0,
                        finish_reason: None,
                    }])
                    .usage(Usage {
                        prompt_tokens: usage.input_tokens,
                        completion_tokens: usage.output_tokens,
                        total_tokens: usage.total_tokens,
                        completion_tokens_details: None,
                        prompt_tokens_details: None,
                    })
                    .build()
            } else {
                tracing::warn!("No usage data in Metadata event");
                ChatCompletionChunkBuilder::new()
                    .id(completion_id.to_string())
                    .created(created_timestamp)
                    .id(completion_id.to_string())
                    .object(CHAT_COMPLETION_OBJECT.to_string())
                    .created(created_timestamp)
                    .model(model.to_string())
                    .choices(vec![ChatCompletionChunkChoice {
                        delta: ChatCompletionChunkChoiceDelta::Content {
                            content: String::new(),
                        },
                        index: 0,
                        finish_reason: None,
                    }])
                    .build() // No usage data
            }
        }
        _ => {
            tracing::warn!("Unknown stream chunk type");
            ChatCompletionChunkBuilder::new()
                .model(model.to_string())
                .choices(vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Content {
                        content: String::new(),
                    },
                    index: 0,
                    finish_reason: None,
                }])
                .build() // Handle unknown types
        }
    };

    create_sse_event(&chunk)
}

/// Creates an SSE event from a chunk
fn create_sse_event(chunk: &ChatCompletionChunk) -> Event {
    let data = serde_json::to_string(&chunk).unwrap_or_else(|e| {
        tracing::error!("Failed to serialize chunk: {}", e);
        String::from("{}")
    });
    Event::default().data(data)
}

async fn chat_completions(
    Json(payload): Json<ChatCompletionsRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let provider = BedrockProvider::new().await;
    provider.chat_completions_stream(payload).await
}
