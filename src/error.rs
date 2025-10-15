use thiserror::Error;

#[derive(Error, Debug)]
pub enum CandleWrapperErrors {
    #[error("failed to load plugin")]
    LoadPluginError,
    #[error("cannot read model config from env:")]
    EnvError(#[from] envy::Error),
    #[error("failed to load config")]
    LoadConfigError,
    #[error("model file(s) not found")]
    ModelNotFound,
    #[error("failed to run")]
    RunError,

    // High-level abstraction errors
    #[error("Model operation failed: {0}")]
    ModelError(String),

    #[error("File operation failed: {0}")]
    FileError(String),

    #[error("Device operation failed: {0}")]
    DeviceError(String),

    // Lower level error wrapping
    #[error("Candle core error: {0}")]
    CandleCoreError(#[from] candle_core::Error),
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),
    #[error("HuggingFace Hub error: {0}")]
    HfHubError(#[from] hf_hub::api::sync::ApiError),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
}

impl CandleWrapperErrors {
    pub fn model_error<T: std::fmt::Display>(msg: T) -> Self {
        Self::ModelError(msg.to_string())
    }

    pub fn file_error<T: std::fmt::Display>(msg: T) -> Self {
        Self::FileError(msg.to_string())
    }

    pub fn device_error<T: std::fmt::Display>(msg: T) -> Self {
        Self::DeviceError(msg.to_string())
    }
}
