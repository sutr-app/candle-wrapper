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

    /// Decoder forward pass that also returns cross-attention weights from specified layers.
    /// Only supported for non-quantized models; returns an error for QWhisper.
    #[inline]
    pub fn decoder_forward_with_cross_attn(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
        alignment_layer_indices: &[usize],
    ) -> candle_core::Result<(Tensor, Vec<(usize, Tensor)>)> {
        match self {
            Self::Whisper(m) => {
                m.decoder
                    .forward_with_cross_attn(x, xa, flush, alignment_layer_indices)
            }
            Self::QWhisper(_) => Err(candle_core::Error::Msg(
                "cross-attention weights extraction is not supported for quantized models"
                    .to_string(),
            )),
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
