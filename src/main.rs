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
    response::sse::{Event, Sse},
    routing::post,
};
use chrono::prelude::*;
use either::Either;
use futures::stream::Stream;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use uuid::Uuid;
use void::Void;

mod good;
use good::{ChatCompletionsRequest, Content, MessageContent, OpenaiMessage, Role};

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
    ) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>;
}

pub struct BedrockProvider {
    client: Client,
}

impl BedrockProvider {
    pub async fn new() -> Self {
        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .region(DEFAULT_AWS_REGION)
            .load()
            .await;
        let client = Client::new(&sdk_config);
        BedrockProvider { client }
    }
}

#[async_trait]
impl ChatProvider for BedrockProvider {
    async fn chat_completions_stream(
        self,
        payload: ChatCompletionsRequest,
    ) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
        let (system_content_blocks, messages) = process_messages(&payload.messages);

        let stream_result = self
            .client
            .converse_stream()
            .model_id(&payload.model)
            .set_system(Some(system_content_blocks))
            .set_messages(Some(messages))
            .send()
            .await;

        let sse_stream = async_stream::stream! {
            let mut stream = match stream_result {
                Ok(response) => response.stream,
                Err(e) => {
                    tracing::error!("Bedrock API error: {}", e);
                    // Return an error event to the client
                    yield Ok(Event::default().data(
                        serde_json::to_string(&ChatCompletionChunkBuilder::new()
                            .model(payload.model.clone())
                            .choices(vec![ChatCompletionChunkChoice {
                                delta: ChatCompletionChunkChoiceDelta::Content {
                                    content: "An error occurred while processing your request.".to_string(),
                                },
                                index: 0,
                                finish_reason: Some("error".to_string()),
                            }])
                            .build()
                        ).unwrap_or_default()
                    ));
                    yield Ok(Event::default().data(SSE_DONE_MESSAGE));
                    return;
                }
            };

            match stream.recv().await {
                Ok(Some(event)) => {
                    if let sse_event = handle_stream_event(&payload.model, event) {
                        yield Ok(sse_event);
                    }
                    while let Ok(Some(event)) = stream.recv().await {
                        if let sse_event = handle_stream_event(&payload.model, event) {
                            yield Ok(sse_event);
                        }
                    }
                },
                Err(e) => {
                    tracing::error!("Stream error: {}", e);
                },
                Ok(None) => {},
            }
            yield Ok(Event::default().data(SSE_DONE_MESSAGE));
        };

        Sse::new(sse_stream)
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

fn process_content(content: &MessageContent) -> Vec<ContentBlock> {
    match content {
        MessageContent::String(text) => vec![ContentBlock::Text(text.clone())],
        MessageContent::Array(contents) => contents
            .iter()
            .map(|c| match c {
                Content::Text { text } => ContentBlock::Text(text.clone()),
            })
            .collect(),
    }
}

fn process_system_content(content: &MessageContent) -> Vec<SystemContentBlock> {
    match content {
        MessageContent::String(text) => vec![SystemContentBlock::Text(text.clone())],
        MessageContent::Array(contents) => contents
            .iter()
            .map(|c| match c {
                Content::Text { text } => SystemContentBlock::Text(text.clone()),
            })
            .collect(),
    }
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
                    .id(Uuid::new_v4().to_string())
                    .object(CHAT_COMPLETION_OBJECT.to_string())
                    .created(Utc::now().timestamp())
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
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let provider = BedrockProvider::new().await;

    provider.chat_completions_stream(payload).await
}
