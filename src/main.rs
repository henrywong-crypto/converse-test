// Constants
// Configuration constants
const DEFAULT_AWS_REGION: &str = "us-east-1";

// Message constants
const CHAT_COMPLETION_OBJECT: &str = "chat.completion.chunk";
const ASSISTANT_ROLE: &str = "assistant";
const SSE_DONE_MESSAGE: &str = "[DONE]";
const STOP_REASON: &str = "stop";

use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::{
    Client,
    types::{ContentBlock, ConversationRole, Message, SystemContentBlock},
};
use axum::response::IntoResponse;
use axum::{
    Json, Router,
    response::sse::{Event, Sse},
    routing::post,
};
use futures::stream::Stream;
use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;
use uuid::Uuid;
use void::Void;

// Error types
#[derive(Debug, Serialize, thiserror::Error)]
pub enum ChatCompletionError {
    #[error("Bedrock API error: {0}")]
    BedrockApi(String),

    #[error("Error receiving stream: {0}")]
    StreamError(String),
}

impl IntoResponse for ChatCompletionError {
    fn into_response(self) -> axum::response::Response {
        let (status, error_message) = match self {
            ChatCompletionError::BedrockApi(msg) => (axum::http::StatusCode::BAD_REQUEST, msg),
            ChatCompletionError::StreamError(msg) => {
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, msg)
            }
        };

        let body = axum::Json(ErrorResponse {
            error: error_message,
        });

        (status, body).into_response()
    }

    #[tokio::test]
    async fn test_chat_completions_success() {
        // This test would require mocking the Bedrock client, which is not possible here.
        // This is a placeholder to show how the test would be structured.

        /*
        let request = ChatCompletionsRequest {
            model: "amazon.titan-text-express-v1".to_string(),
            messages: vec![OpenaiMessage {
                role: Role::User,
                content: MessageContent::String("Hello".to_string()),
            }],
            stream: Some(true),
            ..Default::default() // Add Default trait to ChatCompletionsRequest for easier testing
        };

        let result = chat_completions(Json(request)).await;
        assert!(result.is_ok());

        // Further assertions would be needed to check the content of the SSE stream,
        // which would require awaiting and collecting the events.
        */
    }

    #[tokio::test]
    async fn test_chat_completions_invalid_model() {
        // This test would also require mocking the Bedrock client.
        // This is a placeholder.

        /*
        let request = ChatCompletionsRequest {
            model: "invalid-model-id".to_string(),
            messages: vec![OpenaiMessage {
                role: Role::User,
                content: MessageContent::String("Hello".to_string()),
            }],
            stream: Some(true),
             ..Default::default()
        };

        let result = chat_completions(Json(request)).await;
        assert!(matches!(result, Err(ChatCompletionError::BedrockApi(_))));
        */
    }
}

// Add Default trait to ChatCompletionsRequest
impl Default for ChatCompletionsRequest {
    fn default() -> Self {
        ChatCompletionsRequest {
            model: "".to_string(),
            messages: vec![],
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
        }
    }
}

#[derive(Serialize)]
pub struct ErrorResponse {
    error: String,
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

#[derive(Serialize, Debug)]
pub struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChunkChoice>,
    usage: Option<Usage>,
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let app = Router::new().route("/chat/completions", post(chat_completions));

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

fn process_messages(messages: &[OpenaiMessage]) -> (Vec<SystemContentBlock>, Vec<Message>) {
    let mut system_blocks = Vec::new();
    let mut non_system_messages = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                system_blocks.extend(process_system_content(&msg.content));
            }
            Role::Assistant | Role::User => {
                let content_blocks = process_content(&msg.content);
                let role = match msg.role {
                    Role::Assistant => ConversationRole::Assistant,
                    Role::User => ConversationRole::User,
                    _ => unreachable!(), // We've already handled System above
                };

                if let Ok(message) = Message::builder()
                    .role(role)
                    .set_content(Some(content_blocks))
                    .build()
                {
                    non_system_messages.push(message);
                }
            }
        }
    }

    (system_blocks, non_system_messages)
}

/// Creates a Bedrock client with default configuration
async fn create_bedrock_client() -> Client {
    let sdk_config = aws_config::defaults(BehaviorVersion::latest())
        .region(DEFAULT_AWS_REGION)
        .load()
        .await;
    Client::new(&sdk_config)
}

/// Handles different stream events and creates appropriate chunks
fn handle_stream_event(
    model: &str,
    event: aws_sdk_bedrockruntime::types::ConverseStreamOutput,
) -> Result<Event, ChatCompletionError> {
    match event {
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockDelta(event) => {
            let content = match event.delta {
                Some(aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(text)) => text,
                _ => String::new(),
            };

            let chunk = ChatCompletionChunk {
                id: Uuid::new_v4().to_string(),
                object: CHAT_COMPLETION_OBJECT.to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("SystemTime before Unix epoch")
                    .as_secs(),
                model: model.to_string(),
                choices: vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Content { content },
                    index: 0,
                    finish_reason: None,
                }],
                usage: None,
            };

            create_sse_event(&chunk)
        }
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockStart(_)
        | aws_sdk_bedrockruntime::types::ConverseStreamOutput::MessageStart(_) => {
            let chunk = ChatCompletionChunk {
                id: Uuid::new_v4().to_string(),
                object: CHAT_COMPLETION_OBJECT.to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("SystemTime before Unix epoch")
                    .as_secs(),
                model: model.to_string(),
                choices: vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Role {
                        role: ASSISTANT_ROLE.to_string(),
                    },
                    index: 0,
                    finish_reason: None,
                }],
                usage: None,
            };

            create_sse_event(&chunk)
        }
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockStop(_)
        | aws_sdk_bedrockruntime::types::ConverseStreamOutput::MessageStop(_) => {
            let chunk = ChatCompletionChunk {
                id: Uuid::new_v4().to_string(),
                object: CHAT_COMPLETION_OBJECT.to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("SystemTime before Unix epoch")
                    .as_secs(),
                model: model.to_string(),
                choices: vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Content {
                        content: String::new(),
                    },
                    index: 0,
                    finish_reason: Some(STOP_REASON.to_string()),
                }],
                usage: None,
            };

            create_sse_event(&chunk)
        }
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::Metadata(event) => {
            if let Some(usage) = event.usage {
                let mut chunk = ChatCompletionChunk {
                    id: Uuid::new_v4().to_string(),
                    object: CHAT_COMPLETION_OBJECT.to_string(),
                    created: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("SystemTime before Unix epoch")
                        .as_secs(),
                    model: model.to_string(),
                    choices: vec![ChatCompletionChunkChoice {
                        delta: ChatCompletionChunkChoiceDelta::Content {
                            content: String::new(),
                        },
                        index: 0,
                        finish_reason: None,
                    }],
                    usage: None,
                };
                chunk.usage = Some(Usage {
                    prompt_tokens: usage.input_tokens,
                    completion_tokens: usage.output_tokens,
                    total_tokens: usage.total_tokens,
                    completion_tokens_details: None,
                    prompt_tokens_details: None,
                });
                create_sse_event(&chunk)
            } else {
                let chunk = ChatCompletionChunk {
                    id: Uuid::new_v4().to_string(),
                    object: CHAT_COMPLETION_OBJECT.to_string(),
                    created: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("SystemTime before Unix epoch")
                        .as_secs(),
                    model: model.to_string(),
                    choices: vec![ChatCompletionChunkChoice {
                        delta: ChatCompletionChunkChoiceDelta::Content {
                            content: String::new(),
                        },
                        index: 0,
                        finish_reason: None,
                    }],
                    usage: None,
                };
                // Return a no-op event if there's no usage data
                create_sse_event(&chunk)
            }
        }
        _ => {
            tracing::warn!("Unknown stream chunk type");
            let chunk = ChatCompletionChunk {
                id: Uuid::new_v4().to_string(),
                object: CHAT_COMPLETION_OBJECT.to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("SystemTime before Unix epoch")
                    .as_secs(),
                model: model.to_string(),
                choices: vec![ChatCompletionChunkChoice {
                    delta: ChatCompletionChunkChoiceDelta::Content {
                        content: String::new(),
                    },
                    index: 0,
                    finish_reason: None,
                }],
                usage: None,
            };
            // Return a no-op event for unknown types
            create_sse_event(&chunk)
        }
    }
}

/// Creates an SSE event from a chunk
fn create_sse_event(chunk: &ChatCompletionChunk) -> Result<Event, ChatCompletionError> {
    Ok(Event::default().data(
        serde_json::to_string(&chunk)
            .map_err(|e| ChatCompletionError::StreamError(e.to_string()))?,
    ))
}

/// Main chat completions handler that processes requests and returns SSE streams
async fn chat_completions(
    Json(payload): Json<ChatCompletionsRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, ChatCompletionError>>>, ChatCompletionError> {
    let client = create_bedrock_client().await;
    let (system_content_blocks, messages) = process_messages(&payload.messages);

    let mut stream = client
        .converse_stream()
        .model_id(&payload.model)
        .set_system(Some(system_content_blocks))
        .set_messages(Some(messages))
        .send()
        .await
        .map_err(|e| ChatCompletionError::BedrockApi(e.to_string()))?
        .stream;

    let sse_stream = async_stream::stream! {
        while let Some(event) = stream.recv().await.map_err(|e| ChatCompletionError::StreamError(e.to_string()))? {
            yield handle_stream_event(&payload.model, event);
        }
        yield Ok(Event::default().data(SSE_DONE_MESSAGE));
    };

    Ok(Sse::new(sse_stream))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_string_content() {
        let json_string = r#"{
            "role": "user",
            "content": "Hello, how are you?"
        }"#;

        let message: OpenaiMessage = serde_json::from_str(json_string).unwrap();
        assert!(matches!(message.content, MessageContent::String(_)));
    }

    #[test]
    fn test_deserialize_array_content() {
        let json_array = r#"{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the weather like?"
                }
            ]
        }"#;

        let message: OpenaiMessage = serde_json::from_str(json_array).unwrap();
        assert!(matches!(message.content, MessageContent::Array(_)));
    }

    #[test]
    fn test_deserialize_chat_request_with_string_content() {
        let chat_request = r#"{
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Tell me about Rust."
                }
            ],
            "temperature": 0.7
        }"#;

        let request: ChatCompletionsRequest = serde_json::from_str(chat_request).unwrap();
        assert_eq!(request.model, "gpt-4");
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.temperature, Some(0.7));
    }

    #[test]
    fn test_process_content_string() {
        let content = MessageContent::String("Test content".to_string());
        let result = process_content(&content);
        assert_eq!(result.len(), 1);
        assert!(matches!(&result[0], ContentBlock::Text(text) if text == "Test content"));
    }

    #[test]
    fn test_process_content_array() {
        let content = MessageContent::Array(vec![Content::Text {
            text: "Test content".to_string(),
        }]);
        let result = process_content(&content);
        assert_eq!(result.len(), 1);
        assert!(matches!(&result[0], ContentBlock::Text(text) if text == "Test content"));
    }

    #[test]
    fn test_process_messages_mixed_roles() {
        let messages = vec![
            OpenaiMessage {
                role: Role::System,
                content: MessageContent::String("System instruction".to_string()),
            },
            OpenaiMessage {
                role: Role::User,
                content: MessageContent::String("User message".to_string()),
            },
            OpenaiMessage {
                role: Role::Assistant,
                content: MessageContent::String("Assistant message".to_string()),
            },
        ];

        let (system_blocks, converted_messages) = process_messages(&messages);

        // Check system blocks
        assert_eq!(system_blocks.len(), 1);
        assert!(
            matches!(&system_blocks[0], SystemContentBlock::Text(text) if text == "System instruction")
        );

        // Check converted messages
        assert_eq!(converted_messages.len(), 2);
        assert!(matches!(
            converted_messages[0].role(),
            ConversationRole::User
        ));
        assert!(matches!(
            converted_messages[1].role(),
            ConversationRole::Assistant
        ));
    }

    #[test]
    fn test_deserialize_chat_request_with_array_content() {
        let chat_request_array = r#"{
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Tell me about programming."
                        }
                    ]
                }
            ],
            "temperature": 0.7
        }"#;

        let request: ChatCompletionsRequest = serde_json::from_str(chat_request_array).unwrap();
        assert_eq!(request.model, "gpt-4");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.temperature, Some(0.7));
    }

    #[test]
    fn test_metadata_handling() {
        let model = "test-model";
        let usage = aws_sdk_bedrockruntime::types::TokenUsage::builder()
            .input_tokens(10)
            .output_tokens(20)
            .total_tokens(30)
            .build()
            .unwrap();

        let result = handle_metadata(model, usage).unwrap();
        let data = result.data.unwrap();
        let chunk: ChatCompletionChunk = serde_json::from_str(&data).unwrap();

        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }
}
