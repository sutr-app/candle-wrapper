use anyhow::{Context, Result, anyhow};
use candle_core::Tensor;
use candle_transformers::models::quantized_t5 as qt5;
use candle_transformers::models::t5;
use debug_stub_derive::DebugStub;

#[derive(Debug, Clone)]
pub struct T5Setting {
    pub use_cache: bool,
    pub pad_token_id: usize,
    pub eos_token_id: usize,
    pub decoder_start_token_id: Option<usize>,
}

impl From<&t5::Config> for T5Setting {
    fn from(s: &t5::Config) -> Self {
        T5Setting {
            use_cache: s.use_cache,
            pad_token_id: s.pad_token_id,
            eos_token_id: s.eos_token_id,
            decoder_start_token_id: s.decoder_start_token_id,
        }
    }
}

#[derive(DebugStub, Clone)]
pub enum T5Wrapper {
    Encoder(#[debug_stub = "T5Encoder"] t5::T5EncoderModel, T5Setting),
    ConditionalGenerator(
        #[debug_stub = "T5ConditionalGenerator"] t5::T5ForConditionalGeneration,
        T5Setting,
    ),
    QEncoder(#[debug_stub = "QT5Encoder"] qt5::T5EncoderModel, T5Setting),
    QConditionalGenerator(
        #[debug_stub = "QT5ConditionalGenerator"] qt5::T5ForConditionalGeneration,
        T5Setting,
    ),
}

// for T5 model
impl T5Wrapper {
    // encode
    pub fn encode(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Encoder(m, _) => m.forward(input_ids).map_err(|e| e.into()),
            // TODO check if this is correct
            Self::ConditionalGenerator(m, _) => m
                .encode(input_ids)
                .context("error on encoding by T5ForConditionalGeneration"),
            Self::QEncoder(m, _) => m.forward(input_ids).map_err(|e| e.into()),
            // TODO check if this is correct
            Self::QConditionalGenerator(m, _) => m
                .encode(input_ids)
                .context("error on encoding by T5ForConditionalGeneration"),
            // m => Err(anyhow!("Invalid T5Wrapper variant: {:?}", m)),
        }
    }
    pub fn decode(&mut self, input_ids: &Tensor, decoder_token_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::ConditionalGenerator(m, _) => {
                m.decode(decoder_token_ids, input_ids).map_err(|e| e.into())
            }
            Self::QConditionalGenerator(m, _) => {
                m.decode(decoder_token_ids, input_ids).map_err(|e| e.into())
            }
            _ => Err(anyhow!("Invalid T5Wrapper variant:{:?}", self)),
        }
    }
}
