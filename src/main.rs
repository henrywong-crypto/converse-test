use axum::{Json, Router, routing::post};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;
use void::Void;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenaiRole {
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
    pub role: OpenaiRole,
    #[serde(deserialize_with = "string_or_array")]
    pub content: OpenaiContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OpenaiContent {
    String(String),
    Array(Vec<Content>),
}

impl FromStr for OpenaiContent {
    type Err = Void;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(OpenaiContent::String(s.to_string()))
    }
}

// Example usage and tests
fn main() {
    // Test case 1: String content
    let json_string = r#"{
        "role": "user",
        "content": "Hello, how are you?"
    }"#;

    let message: OpenaiMessage = serde_json::from_str(json_string).unwrap();
    println!("String content: {:?}", message);

    // Test case 2: Array content with Text type
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
    println!("Array content: {:?}", message);

    // Test case 3: Full chat request with string content
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
    println!("Chat request: {:?}", request);

    // Test case 4: Full chat request with array content
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
    println!("Chat request with array content: {:?}", request);

    // Start the server
    start_server();
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

async fn chat_completions(Json(payload): Json<ChatCompletionsRequest>) -> &'static str {
    println!("{:?}", payload);
    "Hello world"
}

#[tokio::main]
async fn start_server() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let app = Router::new().route("/chat/completions", post(chat_completions));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}
