#[cfg(test)]
mod tests {
    use crate::llm::config::{LLMConfig, LLMConfigWrapper, ModelType};
    use serde_json::json;

    /// Test configuration parsing for Qwen3
    #[test]
    fn test_qwen3_config_loading() {
        let config_json = json!({
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 151936,
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "max_position_embeddings": 32768,
            "sliding_window": null,
            "tie_word_embeddings": false,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "bos_token_id": 151643,
            "eos_token_id": 151645
        });

        let llm_config: LLMConfig = serde_json::from_value(config_json).unwrap();
        
        // Test model type detection
        assert_eq!(llm_config.model_type(), Some(ModelType::Qwen3));
        
        // Test conversion to qwen3::Config
        let qwen3_config: candle_transformers::models::qwen3::Config = llm_config.clone().into();
        assert_eq!(qwen3_config.vocab_size, 151936);
        assert_eq!(qwen3_config.hidden_size, 3584);
        assert_eq!(qwen3_config.num_hidden_layers, 28);
        assert_eq!(qwen3_config.head_dim, 3584 / 28); // 128
        assert!(!qwen3_config.attention_bias);
        
        // Test conversion to LLMConfigWrapper
        let wrapper: LLMConfigWrapper = llm_config.into();
        match wrapper {
            LLMConfigWrapper::Qwen3(_) => {
                println!("✅ Qwen3 config wrapper created successfully");
            }
            _ => panic!("Expected Qwen3 config wrapper"),
        }
    }

}