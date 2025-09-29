pub mod config;

use anyhow::Result;
use candle_core::{Tensor, IndexOp};
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
    Qwen3(qwen3::ModelForCausalLM),            // SafeTensors version with lm_head
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
        
        // Try to get hidden states directly instead of logits
        let hidden_states = self.get_hidden_states(input_ids, 0, None)?;
        let dims = hidden_states.dims();
        
        if dims.len() != 3 {
            return Err(anyhow::anyhow!("Expected 3D tensor [batch, seq, hidden], got {:?}", dims).into());
        }
        
        let (_batch_size, seq_len, _hidden_dim) = hidden_states.dims3()?;
        
        // Extract the last token's hidden state for embedding
        // Following gte-qwen example: narrow to last position, then squeeze
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)?;
        let squeezed = last_hidden.squeeze(1)?; // Remove seq dimension
        
        // Ensure F32 dtype for consistency
        Ok(squeezed.to_dtype(DType::F32)?)
    }

    /// Get hidden states without applying lm_head (for embedding generation)  
    pub fn get_hidden_states(&mut self, input_ids: &Tensor, seqlen_offset: usize, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        match self {
            Self::QQwen3(_m) => {
                // GGUF quantized models have lm_head applied, not suitable for embeddings
                return Err(anyhow::anyhow!(
                    "GGUF quantized models apply lm_head and return logits. Use SafeTensors embedding models instead."
                ));
            },
            Self::Qwen3(m) => {
                // SafeTensors version of embedding model should return hidden states directly
                let output = m.forward(input_ids, seqlen_offset)?;
                // Verify the output shape is reasonable for hidden states
                let dims = output.dims();
                if dims.len() == 3 {
                    // Expected format: [batch_size, seq_len, hidden_dim]
                    Ok(output)
                } else if dims.len() == 2 && dims[1] < 10000 {
                    // Might be [batch_size, hidden_dim], add sequence dimension
                    let output_3d = output.unsqueeze(1)?; // Add seq_len=1 dimension
                    Ok(output_3d)
                } else {
                    return Err(anyhow::anyhow!(
                        "Unexpected output shape: {:?}. Expected 3D hidden states or 2D embeddings",
                        dims
                    ));
                }
            },
            _ => {
                // For other models, try standard forward
                let output = self.forward(input_ids, seqlen_offset)?;
                Ok(output)
            }
        }
    }

    /// Generate embeddings for batch processing
    /// Returns tensor of shape [batch_size, hidden_dim]
    pub fn generate_embeddings_batch(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        // This is essentially the same as generate_embedding but more explicit about batch handling
        self.generate_embedding(input_ids)
    }

    /// Generate single embedding from batch (takes first item)
    /// Returns tensor of shape [hidden_dim]
    pub fn generate_single_embedding(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let batch_embeddings = self.generate_embedding(input_ids)?;
        
        // If batch size is 1, squeeze batch dimension
        if batch_embeddings.dims()[0] == 1 {
            Ok(batch_embeddings.squeeze(0)?)
        } else {
            // Take first embedding from batch
            Ok(batch_embeddings.narrow(0, 0, 1)?.squeeze(0)?)
        }
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

    /// Compute reranking score for single query-document pair
    /// Uses yes/no token logits for binary relevance scoring
    pub fn compute_reranking_score(
        &mut self,
        input_ids: &Tensor,
        yes_token_id: u32,
        no_token_id: u32,
    ) -> Result<f32> {
        // Forward pass to get logits
        let logits = self.forward(input_ids, 0)?;

        // Extract last token logits for classification
        let dims = logits.dims();
        if dims.len() != 3 {
            return Err(anyhow::anyhow!("Expected 3D logits tensor [batch, seq, vocab], got {:?}", dims).into());
        }

        let (_batch_size, seq_len, vocab_size) = (dims[0], dims[1], dims[2]);


        // Validate token IDs are within vocabulary bounds
        if yes_token_id as usize >= vocab_size {
            return Err(anyhow::anyhow!("yes_token_id {} out of vocabulary bounds {}", yes_token_id, vocab_size).into());
        }
        if no_token_id as usize >= vocab_size {
            return Err(anyhow::anyhow!("no_token_id {} out of vocabulary bounds {}", no_token_id, vocab_size).into());
        }

        // Extract last token's logits
        let last_logits = logits.narrow(1, seq_len - 1, 1)?; // [batch, 1, vocab]

        // Get yes/no token logits with bounds checking
        let yes_logit = last_logits
            .i((0, 0, yes_token_id as usize))?
            .to_scalar::<f32>()?;
        let no_logit = last_logits
            .i((0, 0, no_token_id as usize))?
            .to_scalar::<f32>()?;

        // Apply softmax to get probability
        let yes_exp = yes_logit.exp();
        let no_exp = no_logit.exp();
        let score = yes_exp / (yes_exp + no_exp);

        Ok(score)
    }

    /// Compute reranking scores for batch of query-document pairs
    /// Returns vector of scores corresponding to each input in the batch
    pub fn batch_compute_reranking_scores(
        &mut self,
        batch_input_ids: &Tensor,
        yes_token_id: u32,
        no_token_id: u32,
    ) -> Result<Vec<f32>> {
        // Forward pass to get batch logits
        let logits = self.forward(batch_input_ids, 0)?;

        let dims = logits.dims();
        if dims.len() != 3 {
            return Err(anyhow::anyhow!("Expected 3D logits tensor [batch, seq, vocab], got {:?}", dims).into());
        }

        let (batch_size, seq_len, vocab_size) = (dims[0], dims[1], dims[2]);


        // Validate token IDs are within vocabulary bounds
        if yes_token_id as usize >= vocab_size {
            return Err(anyhow::anyhow!("yes_token_id {} out of vocabulary bounds {}", yes_token_id, vocab_size).into());
        }
        if no_token_id as usize >= vocab_size {
            return Err(anyhow::anyhow!("no_token_id {} out of vocabulary bounds {}", no_token_id, vocab_size).into());
        }

        let mut scores = Vec::with_capacity(batch_size);

        // Process each sample in the batch
        for batch_idx in 0..batch_size {
            // Get last token logits for this sample
            let sample_logits = logits.i((batch_idx, seq_len - 1, ..))?; // [vocab]

            // Extract yes/no token logits with bounds checking
            let yes_logit = sample_logits.i(yes_token_id as usize)?.to_scalar::<f32>()?;
            let no_logit = sample_logits.i(no_token_id as usize)?.to_scalar::<f32>()?;

            // Apply softmax to get probability
            let yes_exp = yes_logit.exp();
            let no_exp = no_logit.exp();
            let score = yes_exp / (yes_exp + no_exp);

            scores.push(score);
        }

        Ok(scores)
    }

    /// Validate that model supports reranking by checking token vocabulary
    /// Returns (yes_token_id, no_token_id) if found, error otherwise
    pub fn validate_reranking_tokens(&self, tokenizer: &tokenizers::Tokenizer) -> Result<(u32, u32)> {
        let yes_token_id = tokenizer
            .token_to_id("yes")
            .or_else(|| tokenizer.token_to_id("Yes"))
            .or_else(|| tokenizer.token_to_id("YES"))
            .ok_or_else(|| anyhow::anyhow!("'yes' token not found in vocabulary"))?;

        let no_token_id = tokenizer
            .token_to_id("no")
            .or_else(|| tokenizer.token_to_id("No"))
            .or_else(|| tokenizer.token_to_id("NO"))
            .ok_or_else(|| anyhow::anyhow!("'no' token not found in vocabulary"))?;

        Ok((yes_token_id, no_token_id))
    }
}
