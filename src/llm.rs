pub mod config;

use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::falcon;
use candle_transformers::models::llama;
use candle_transformers::models::mistral;
use candle_transformers::models::qwen3;
use candle_transformers::models::quantized_qwen3;
use candle_transformers::models::stable_lm;

#[derive(Debug)]
pub enum LLMModelWrapper {
    Falcon(falcon::Falcon),
    Llama(llama::Llama, llama::Cache),
    Mistral(mistral::Model),
    StableLM(stable_lm::Model),
    Qwen3(qwen3::Model),                       // SafeTensors version
    QQwen3(quantized_qwen3::ModelWeights),    // GGUF quantized version
    // QStableLM(quantized_stable_lm::Model),
}
impl LLMModelWrapper {
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        match self {
            // TODO not working?
            Self::Falcon(m) => m.forward(input_ids).map_err(|e| e.into()),
            Self::Llama(m, c) => m.forward(input_ids, seqlen_offset, c).map_err(|e| e.into()),
            Self::Mistral(m) => m.forward(input_ids, seqlen_offset).map_err(|e| e.into()),
            Self::StableLM(m) => m.forward(input_ids, seqlen_offset).map_err(|e| e.into()),
            Self::Qwen3(m) => {
                // SafeTensors version of Qwen3
                m.forward(input_ids, seqlen_offset).map_err(|e| e.into())
            },
            Self::QQwen3(m) => {
                // GGUF quantized version of Qwen3
                m.forward(input_ids, seqlen_offset).map_err(|e| e.into())
            },
            // Self::QStableLM(_) => todo!(),
        }
    }

    /// Generate embedding using the last token's hidden state
    pub fn generate_embedding(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        use candle_core::DType;
        
        let seq_len = input_ids.dims()[1];
        let logits = self.forward(input_ids, 0)?;
        
        // Extract the last token's hidden state for embedding
        let last_hidden = logits.narrow(1, seq_len - 1, 1)?;
        let squeezed = last_hidden.squeeze(1)?;
        Ok(squeezed.to_dtype(DType::F32)?)
    }
    
    /// Get model type for debugging/logging
    pub fn model_type(&self) -> &'static str {
        match self {
            Self::Falcon(_) => "Falcon",
            Self::Llama(_, _) => "Llama",
            Self::Mistral(_) => "Mistral", 
            Self::StableLM(_) => "StableLM",
            Self::Qwen3(_) => "Qwen3-SafeTensors",
            Self::QQwen3(_) => "Qwen3-GGUF",
        }
    }
}
