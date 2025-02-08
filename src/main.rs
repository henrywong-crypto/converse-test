use crate::constants::*;
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::{
    self, Client,
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

mod constants;
mod good;
use good::{
    ChatCompletionsRequest, Content, MessageContent, OpenaiMessage, Role, process_content,
    process_system_content,
};

// Make our own error that wraps `anyhow::Error`.
struct AppError(anyhow::Error);

// Tell axum how to convert `AppError` into a response.
impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", self.0),
        )
            .into_response()
    }
}

// This enables using `?` on functions that return `Result<_, anyhow::Error>` to turn them into
// `Result<_, AppError>`. That way you don't need to do that manually.
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

#[derive(Serialize, Debug, Default)]
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

impl ChatCompletionChunkBuilder {
    pub fn new() -> Self {
        ChatCompletionChunkBuilder {
            chunk: ChatCompletionChunk::default(),
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
        self.chunk.usage = Some(usage); // Keep this as Some(usage) for now
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
    ) -> anyhow::Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>>;
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
    ) -> anyhow::Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>> {
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
            .map_err(|e| anyhow::anyhow!("Bedrock API error: {}", e))?
            .stream;

        let sse_stream = async_stream::stream! {
            loop {
                match stream.recv().await {
                    Ok(Some(event)) => {
                        match handle_stream_event(&payload.model, &completion_id, created_timestamp, event) {
                            Ok(sse_event) => yield Ok(sse_event),
                            Err(e) => {
                                tracing::error!("Error handling stream event: {}", e);
                                // Attempt to send an error chunk
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
                                // If serialization fails here, there's not much we can do, but at least log it
                                if let Ok(sse_event) = create_sse_event(&error_chunk) {
                                    yield Ok(sse_event);
                                }
                                break;
                            }
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
                        // If serialization fails here, there's not much we can do
                        if let Ok(sse_event) = create_sse_event(&error_chunk) {
                            yield Ok(sse_event);
                        }
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
    messages.iter().fold(
        (Vec::new(), Vec::new()),
        |(mut system_blocks, mut conversation_messages), msg| {
            match msg.role {
                Role::System => {
                    system_blocks.extend(process_system_content(&msg.content));
                }
                Role::Assistant | Role::User => {
                    if let Some(message) = process_conversation_message(msg) {
                        conversation_messages.push(message);
                    }
                }
            }
            (system_blocks, conversation_messages)
        },
    )
}

/// Handles different stream events and creates appropriate chunks
fn handle_stream_event(
    model: &str,
    completion_id: &str,
    created_timestamp: i64,
    event: aws_sdk_bedrockruntime::types::ConverseStreamOutput,
) -> anyhow::Result<Event> {
    // Initialize builder with common fields
    let mut builder = ChatCompletionChunkBuilder::new()
        .id(completion_id.to_string())
        .object(CHAT_COMPLETION_OBJECT.to_string())
        .created(created_timestamp)
        .model(model.to_string());

    builder =
        match event {
            aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockDelta(event) => {
                let content = event
                    .delta
                    .and_then(|d| match d {
                        aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(text) => Some(text),
                        _ => None,
                    })
                    .unwrap_or_default();

                builder.choices(vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Content { content },
                    index: 0,
                    finish_reason: None,
                }])
            }
            aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockStart(_)
            | aws_sdk_bedrockruntime::types::ConverseStreamOutput::MessageStart(_) => builder
                .choices(vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Role {
                        role: ASSISTANT_ROLE.to_string(),
                    },
                    index: 0,
                    finish_reason: None,
                }]),
            aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockStop(_)
            | aws_sdk_bedrockruntime::types::ConverseStreamOutput::MessageStop(_) => builder
                .choices(vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Content {
                        content: String::new(),
                    },
                    index: 0,
                    finish_reason: Some(STOP_REASON.to_string()),
                }]),
            aws_sdk_bedrockruntime::types::ConverseStreamOutput::Metadata(event) => {
                let base_builder = builder.choices(vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Content {
                        content: String::new(),
                    },
                    index: 0,
                    finish_reason: None,
                }]);

                if let Some(usage) = event.usage {
                    base_builder.usage(Usage {
                        prompt_tokens: usage.input_tokens,
                        completion_tokens: usage.output_tokens,
                        total_tokens: usage.total_tokens,
                        completion_tokens_details: None,
                        prompt_tokens_details: None,
                    })
                } else {
                    tracing::warn!("No usage data in Metadata event");
                    base_builder
                }
            }
            _ => {
                tracing::warn!("Unknown stream chunk type");
                builder.choices(vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Content {
                        content: String::new(),
                    },
                    index: 0,
                    finish_reason: None,
                }])
            }
        };

    let chunk = builder.build();

    create_sse_event(&chunk)
}

/// Creates an SSE event from a chunk
fn create_sse_event(chunk: &ChatCompletionChunk) -> anyhow::Result<Event> {
    serde_json::to_string(&chunk)
        .map(|data| Event::default().data(data))
        .map_err(|e| anyhow::anyhow!("Serialization error: {}", e))
}

async fn chat_completions(
    Json(payload): Json<ChatCompletionsRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, AppError> {
    let provider = BedrockProvider::new().await;
    Ok(provider.chat_completions_stream(payload).await?)
}
