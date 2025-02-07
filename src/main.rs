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
use either::Either;
use futures::stream::Stream;
use itertools::Itertools;
use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;
use uuid::Uuid;
use void::Void;

use chrono::prelude::*;

// Error types
#[derive(Debug, Serialize, thiserror::Error)]
pub enum ChatCompletionError {
    #[error("Bedrock API error: {0}")]
    BedrockApi(String),

    #[error("Error receiving stream: {0}")]
    StreamError(String),
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Assistant,
    User,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum Content {
    #[serde(rename = "text")]
    Text { text: String },
}

#[derive(Debug, Deserialize)]
pub struct OpenaiMessage {
    pub role: Role,
    #[serde(deserialize_with = "string_or_array")]
    pub content: MessageContent,
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

    pub fn usage(mut self, usage: Option<Usage>) -> Self {
        self.chunk.usage = usage;
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

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    String(String),
    Array(Vec<Content>),
}

impl FromStr for MessageContent {
    type Err = Void;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(MessageContent::String(s.to_string()))
    }
}

#[async_trait]
pub trait ChatProvider {
    async fn chat_completions_stream(
        self,
        request: ChatCompletionsRequest,
    ) -> Result<Sse<impl Stream<Item = Result<Event, ChatCompletionError>>>, ChatCompletionError>;
}

#[derive(Clone)]
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
    ) -> Result<Sse<impl Stream<Item = Result<Event, ChatCompletionError>>>, ChatCompletionError>
    {
        let (system_content_blocks, messages) = process_messages(&payload.messages);

        let mut stream = self
            .client
            .converse_stream()
            .model_id(&payload.model)
            .set_system(Some(system_content_blocks))
            .set_messages(Some(messages))
            .send()
            .await
            .map_err(|e| ChatCompletionError::BedrockApi(e.to_string()))?
            .stream;

        let sse_stream = async_stream::stream! {
            match stream.recv().await {
                Ok(Some(event)) => {
                    match handle_stream_event(&payload.model, event) {
                        Ok(sse_event) => yield Ok(sse_event),
                        Err(e) => yield Err(e),
                    }
                    while let Some(event) = stream.recv().await.map_err(|e| ChatCompletionError::StreamError(e.to_string()))? {
                        match handle_stream_event(&payload.model, event) {
                            Ok(sse_event) => yield Ok(sse_event),
                            Err(e) => yield Err(e),
                        }
                    }

                },
                Err(e) => yield Err(ChatCompletionError::StreamError(e.to_string())),
                Ok(None) => {},
            }
            yield Ok(Event::default().data(SSE_DONE_MESSAGE));
        };

        Ok(Sse::new(sse_stream))
    }
}

const DEFAULT_AWS_REGION: &str = "us-east-1";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let bedrock_provider = BedrockProvider::new().await;

    let app = Router::new().route(
        "/chat/completions",
        post(move |json| chat_completions(json, bedrock_provider)),
    );

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsRequest {
    /// ID of the model to use.
    /// See the model endpoint compatibility table for details on which models work with the Chat API.
    pub model: String,
    /// The messages to generate chat completions for, in the chat format.
    pub messages: Vec<OpenaiMessage>,
    /// What sampling temperature to use, between 0 and 2.
    /// Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    /// We generally recommend altering this or top_p but not both.
    /// Defaults to 1
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// An alternative to sampling with temperature, called nucleus sampling,
    /// where the model considers the results of the tokens with top_p probability mass.
    /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    /// We generally recommend altering this or temperature but not both.
    /// Defaults to 1
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// How many chat completion choices to generate for each input message.
    /// Defaults to 1
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i32>,
    /// If set, partial message deltas will be sent, like in ChatGPT.
    /// Tokens will be sent as data-only server-sent events as they become available,
    /// with the stream terminated by a data: [DONE] message. See the OpenAI Cookbook for example code.
    /// Defaults to false
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Up to 4 sequences where the API will stop generating further tokens.
    /// Defaults to null
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// The maximum number of tokens to generate in the chat completion.
    /// The total length of input tokens and generated tokens is limited by the model's context length.
    /// Defaults to inf
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    /// Number between -2.0 and 2.0.
    /// Positive values penalize new tokens based on whether they appear in the text so far,
    /// increasing the model's likelihood to talk about new topics.
    /// Defaults to 0
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Number between -2.0 and 2.0.
    /// Positive values penalize new tokens based on their existing frequency in the text so far,
    /// decreasing the model's likelihood to repeat the same line verbatim.
    /// Defaults to 0
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Modify the likelihood of specified tokens appearing in the completion.
    /// Accepts a json object that maps tokens (specified by their token ID in the tokenizer)
    /// to an associated bias value from -100 to 100. Mathematically,
    /// the bias is added to the logits generated by the model prior to sampling.
    /// The exact effect will vary per model, but values between -1 and 1 should
    /// decrease or increase likelihood of selection;
    /// values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    /// Defaults to null
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, String>>,
    /// A unique identifier representing your end-user,
    /// which can help OpenAI to monitor and detect abuse. Learn more.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

fn string_or_array<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Deserialize<'de> + FromStr<Err = Void>,
    D: Deserializer<'de>,
{
    struct StringOrArray<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for StringOrArray<T>
    where
        T: Deserialize<'de> + FromStr<Err = Void>,
    {
        type Value = T;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("string or array")
        }

        fn visit_str<E>(self, value: &str) -> Result<T, E>
        where
            E: de::Error,
        {
            Ok(FromStr::from_str(value).expect("FromStr implementation for void cannot fail"))
        }

        fn visit_seq<S>(self, seq: S) -> Result<T, S::Error>
        where
            S: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(StringOrArray(PhantomData))
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
        conversation_messages
            .into_iter()
            .filter_map(|x| x)
            .collect(),
    )
}

/// Handles different stream events and creates appropriate chunks
fn handle_stream_event(
    model: &str,
    event: aws_sdk_bedrockruntime::types::ConverseStreamOutput,
) -> Result<Event, ChatCompletionError> {
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
                .id(Uuid::new_v4().to_string())
                .object(CHAT_COMPLETION_OBJECT.to_string())
                .created(Utc::now().timestamp())
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
                .id(Uuid::new_v4().to_string())
                .object(CHAT_COMPLETION_OBJECT.to_string())
                .created(Utc::now().timestamp())
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
                .id(Uuid::new_v4().to_string())
                .object(CHAT_COMPLETION_OBJECT.to_string())
                .created(Utc::now().timestamp())
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
                    .usage(Some(Usage {
                        prompt_tokens: usage.input_tokens,
                        completion_tokens: usage.output_tokens,
                        total_tokens: usage.total_tokens,
                        completion_tokens_details: None,
                        prompt_tokens_details: None,
                    }))
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
                .build() // Handle unknown types
        }
    };

    create_sse_event(&chunk)
}

/// Creates an SSE event from a chunk
fn create_sse_event(chunk: &ChatCompletionChunk) -> Result<Event, ChatCompletionError> {
    Ok(Event::default().data(
        serde_json::to_string(&chunk)
            .map_err(|e| ChatCompletionError::StreamError(e.to_string()))?,
    ))
}

async fn chat_completions(
    Json(payload): Json<ChatCompletionsRequest>,
    provider: BedrockProvider,
) -> Sse<impl Stream<Item = Result<Event, ChatCompletionError>>> {
    provider.chat_completions_stream(payload).await.unwrap()
}
