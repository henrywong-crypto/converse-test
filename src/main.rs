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

#[derive(Debug, Serialize, thiserror::Error)]
pub enum ChatCompletionError {
    #[error("Bedrock API error: {0}")]
    BedrockApi(String),

    #[error("Error receiving stream: {0}")]
    StreamError(String),

    #[error("Utterance not found in stream chunk")]
    UtteranceNotFound,

    #[error("Internal Server Error")]
    InternalServerError,
}

impl IntoResponse for ChatCompletionError {
    fn into_response(self) -> axum::response::Response {
        let (status, error_message) = match self {
            ChatCompletionError::BedrockApi(msg) => (axum::http::StatusCode::BAD_REQUEST, msg),
            ChatCompletionError::StreamError(msg) => {
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, msg)
            }
            ChatCompletionError::UtteranceNotFound => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "Utterance not found".to_string(),
            ),
            ChatCompletionError::InternalServerError => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "Internal Server Error".to_string(),
            ),
        };

        let body = axum::Json(ErrorResponse {
            error: error_message,
        });

        (status, body).into_response()
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
            Ok(FromStr::from_str(value).unwrap())
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
            .map(|c| {
                let Content::Text { text } = c;
                ContentBlock::Text(text.clone())
            })
            .collect(),
    }
}

fn process_system_content(content: &MessageContent) -> Vec<SystemContentBlock> {
    match content {
        MessageContent::String(text) => vec![SystemContentBlock::Text(text.clone())],
        MessageContent::Array(contents) => contents
            .iter()
            .map(|c| {
                let Content::Text { text } = c;
                SystemContentBlock::Text(text.clone())
            })
            .collect(),
    }
}

fn process_messages(messages: &[OpenaiMessage]) -> (Vec<SystemContentBlock>, Vec<Message>) {
    let (system_messages, non_system_messages): (Vec<_>, Vec<_>) = messages
        .iter()
        .partition(|msg| matches!(msg.role, Role::System));

    let system_blocks = system_messages
        .iter()
        .flat_map(|msg| process_system_content(&msg.content))
        .collect();

    let non_system_messages = non_system_messages
        .iter()
        .filter_map(|msg| {
            let content_blocks = process_content(&msg.content);

            let role = match msg.role {
                Role::System => {
                    tracing::warn!("System role encountered in non-system messages");
                    None
                }
                Role::Assistant => Some(ConversationRole::Assistant),
                Role::User => Some(ConversationRole::User),
            }?;

            Message::builder()
                .role(role)
                .set_content(Some(content_blocks))
                .build()
                .ok()
        })
        .collect();

    (system_blocks, non_system_messages)
}

// Helper function to create a chat completion chunk
fn create_chunk(
    model: &str,
    role_or_content: ChatCompletionChunkChoiceDelta,
    finish_reason: Option<String>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: Uuid::new_v4().to_string(),
        object: "chat.completion.chunk".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: model.to_string(),
        choices: vec![ChatCompletionChunkChoice {
            delta: role_or_content,
            index: 0,
            finish_reason,
        }],
    }
}

// Main chat completions handler
async fn chat_completions(
    Json(payload): Json<ChatCompletionsRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, ChatCompletionError>>>, ChatCompletionError> {
    let sdk_config = aws_config::defaults(BehaviorVersion::latest())
        .region("us-east-1")
        .load()
        .await;

    let client = Client::new(&sdk_config);

    // Create a model_id variable as requested by the user.
    let model_id = &payload.model;

    // Convert each OpenaiMessage to aws_sdk_bedrockruntime::types::Message
    let (system_content_blocks, messages) = process_messages(&payload.messages);

    let response_stream = client
        .converse_stream()
        .model_id(model_id)
        .set_system(Some(system_content_blocks))
        .set_messages(Some(messages))
        .send()
        .await;

    let mut stream = match response_stream {
        Ok(output) => output.stream,
        Err(e) => return Err(ChatCompletionError::BedrockApi(e.to_string())),
    };

    let sse_stream = async_stream::stream! {
        while let Some(part) = stream.recv().await.map_err(|e| ChatCompletionError::StreamError(e.to_string()))? {
            match part {
                aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockDelta(event) => {
                    let content = match event.delta {
                        Some(aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(text)) => text,
                        _ => String::new(),
                    };
                    let chunk = create_chunk(
                        &payload.model,
                        ChatCompletionChunkChoiceDelta::Content { content },
                        None
                    );
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                },
                aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockStart(_) |
                aws_sdk_bedrockruntime::types::ConverseStreamOutput::MessageStart(_) => {
                    let chunk = create_chunk(
                        &payload.model,
                        ChatCompletionChunkChoiceDelta::Role {
                            role: "assistant".to_string(),
                        },
                        None
                    );
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                },
                aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockStop(_) |
                aws_sdk_bedrockruntime::types::ConverseStreamOutput::MessageStop(_) => {
                    let chunk = create_chunk(
                        &payload.model,
                        ChatCompletionChunkChoiceDelta::Content {
                            content: String::new(),
                        },
                        Some("stop".to_string())
                    );
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                },
                aws_sdk_bedrockruntime::types::ConverseStreamOutput::Metadata(_event) => {
                    // We can ignore metadata events for now
                    continue;
                },
                _ => {
                    tracing::warn!("Unknown stream chunk type");
                    continue;
                }
            }
        }
        yield Ok(Event::default().data("[DONE]"));
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
}
