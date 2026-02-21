use candle_core::Tensor;
use candle_transformers::models::whisper::{self as m, Config};

#[derive(Debug, Clone)]
pub enum WhisperWrapper {
    Whisper(m::model::Whisper),
    QWhisper(m::quantized_model::Whisper),
}

// Maybe we should use some traits rather than doing the dispatch for all these.
impl WhisperWrapper {
    #[inline]
    pub fn config(&self) -> &Config {
        match self {
            Self::Whisper(m) => &m.config,
            Self::QWhisper(m) => &m.config,
        }
    }

    #[inline]
    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Whisper(m) => m.encoder.forward(x, flush),
            Self::QWhisper(m) => m.encoder.forward(x, flush),
        }
    }

    #[inline]
    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Whisper(m) => m.decoder.forward(x, xa, flush),
            Self::QWhisper(m) => m.decoder.forward(x, xa, flush),
        }
    }

    #[inline]
    pub fn decoder_final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Whisper(m) => m.decoder.final_linear(x),
            Self::QWhisper(m) => m.decoder.final_linear(x),
        }
    }

    #[inline]
    pub fn reset_kv_cache(&mut self) {
        match self {
            Self::Whisper(m) => m.reset_kv_cache(),
            Self::QWhisper(m) => m.reset_kv_cache(),
        }
    }
}
