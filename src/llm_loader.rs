use anyhow::{anyhow, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::{qwen3, quantized_qwen3};
use candle_core::quantized::gguf_file;
use std::path::{Path, PathBuf};

use crate::llm::config::{LLMConfig, LLMConfigWrapper, ModelType};
use crate::llm::LLMModelWrapper;
use crate::loader::HFModelLoader;
use crate::models::{ConfigWrapper, ModelWrapper};

/// LLM Model Loader Implementation
pub struct LLMLoaderImpl {
    pub model_id: String,
    pub tokenizer_model_id: Option<String>, // For GGUF models, tokenizer may come from a different model
    pub use_cpu: bool,
    pub model_type: ModelType,
    pub max_seq_length: usize,
    pub model_files: Vec<String>,
}

impl LLMLoaderImpl {
    pub fn new(
        model_id: String,
        tokenizer_model_id: Option<String>,
        use_cpu: bool,
        model_type: ModelType,
        max_seq_length: usize,
        model_files: Vec<String>,
    ) -> Self {
        Self {
            model_id,
            tokenizer_model_id,
            use_cpu,
            model_type,
            max_seq_length,
            model_files,
        }
    }
}

impl LLMLoaderImpl {
    /// Create a simplified loader for non-GGUF models
    pub fn new_simple(
        model_id: String,
        use_cpu: bool,
        model_type: ModelType,
        max_seq_length: usize,
    ) -> Self {
        Self::new(model_id, None, use_cpu, model_type, max_seq_length, vec![])
    }
    
    /// Create a loader for GGUF models with separate tokenizer
    pub fn new_gguf(
        model_id: String,
        tokenizer_model_id: String,
        use_cpu: bool,
        model_type: ModelType,
        max_seq_length: usize,
        model_files: Vec<String>,
    ) -> Self {
        Self::new(model_id, Some(tokenizer_model_id), use_cpu, model_type, max_seq_length, model_files)
    }
}

impl HFModelLoader for LLMLoaderImpl {
    fn load_config_inner(&self, config_file: &Path) -> Result<ConfigWrapper> {
        let config_data = std::fs::read(config_file)?;
        let llm_config: LLMConfig = serde_json::from_slice(&config_data)?;
        Ok(ConfigWrapper::LLM(llm_config.into()))
    }
    
    /// Override load_from_setting to handle GGUF models properly
    fn load_from_setting(&self, setting: crate::models::HFModelSetting) -> Result<crate::models::Models> {
        use crate::models::{Models, TOKENIZER_FILE, MODEL_CONFIG_FILE, WEIGHT_FILES_SAFETENSOR_SINGLE};
        use tokenizers::Tokenizer;
        
        let tokenizer_filename = std::path::PathBuf::from(
            setting.tokenizer_file.as_deref().unwrap_or(TOKENIZER_FILE)
        );
        let mfile = std::path::PathBuf::from(
            setting.model_config_file.as_deref().unwrap_or(MODEL_CONFIG_FILE)
        );
        
        let filenames = setting
            .weight_files
            .as_ref()
            .unwrap_or(
                &WEIGHT_FILES_SAFETENSOR_SINGLE
                    .map(|s| s.to_string())
                    .as_ref()
                    .to_vec(),
            )
            .iter()
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>();
            
        // Check if this is a GGUF model by file extension
        let is_gguf_model = filenames.first()
            .and_then(|f| f.extension())
            .and_then(|ext| ext.to_str())
            .map(|ext| ext == "gguf")
            .unwrap_or(false);
            
        // For GGUF models with separate tokenizer, load tokenizer from different model
        let tokenizer = if let Some(tokenizer_model_id) = &self.tokenizer_model_id {
            tracing::info!("Loading tokenizer from separate model: {}", tokenizer_model_id);
            
            // Load tokenizer from the specified tokenizer model using standard filename
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.repo(hf_hub::Repo::with_revision(
                tokenizer_model_id.clone(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            ));
            // Always use standard tokenizer.json filename for separate tokenizer model
            let tokenizer_path = repo.get("tokenizer.json")?;
            
            Tokenizer::from_file(tokenizer_path).map_err(|e| {
                anyhow::anyhow!("error in load tokenizer from {}: {:?}", tokenizer_model_id, e)
            })?
        } else {
            // Load model files from HF or local for standard tokenizer
            let (tf, _, _) = if let Some(mid) = setting.model_id.as_ref() {
                tracing::info!("load model with hugginface: {}", &mid);
                Self::load_files_from_repository(
                    mid,
                    setting.revision.as_ref(),
                    &tokenizer_filename,
                    &mfile,
                    &filenames,
                )
            } else {
                tracing::info!("load model with localfile: {}", &mfile.to_string_lossy());
                Ok((tokenizer_filename.clone(), mfile.clone(), filenames.clone()))
            }?;
            
            // Use tokenizer from same model (standard case)
            Tokenizer::from_file(&tf).map_err(|e| {
                anyhow::anyhow!("error in load tokenizer {:?}: {:?}", &tf, e)
            })?
        };
        
        // Load weight files and handle config differently for GGUF vs SafeTensors
        let (model, device) = if is_gguf_model {
            tracing::info!("Loading GGUF model - skipping external config.json");
            
            // For GGUF models: load weights directly, config is embedded in GGUF
            let wf = if let Some(mid) = setting.model_id.as_ref() {
                tracing::info!("load GGUF model weights from: {}", &mid);
                let api = hf_hub::api::sync::Api::new()?;
                let repo = api.repo(hf_hub::Repo::with_revision(
                    mid.clone(),
                    hf_hub::RepoType::Model,
                    "main".to_string(),
                ));
                
                let mut weight_files = Vec::new();
                for filename in &filenames {
                    let weight_file = repo.get(&filename.as_os_str().to_string_lossy())?;
                    weight_files.push(weight_file);
                }
                weight_files
            } else {
                filenames
            };
            
            // Load GGUF model directly without external config
            self.load_gguf_model_direct(&wf)
        } else {
            // Standard SafeTensors case: use automatic SafeTensors file detection
            let (config, wf) = if let Some(mid) = setting.model_id.as_ref() {
                tracing::info!("Loading SafeTensors model from HuggingFace: {}", &mid);
                
                let api = hf_hub::api::sync::Api::new()?;
                let repo = api.repo(hf_hub::Repo::with_revision(
                    mid.clone(),
                    hf_hub::RepoType::Model,
                    setting.revision.as_deref().unwrap_or("main").to_string(),
                ));
                
                // Load config from repository
                let config_path = repo.get(&mfile.as_os_str().to_string_lossy())?;
                let config = self.load_config_inner(&config_path)?;
                
                // Load SafeTensors files automatically (index.json or single file)
                let wf = self.load_safetensors_files_auto(&repo)?;
                
                (config, wf)
            } else {
                tracing::info!("Loading SafeTensors model from local files: {}", &mfile.to_string_lossy());
                let config = self.load_config_inner(&mfile)?;
                (config, filenames)
            };
            
            tracing::debug!("Loading SafeTensors with {} weight files", wf.len());
            self.load_model_inner(&config, &wf)
        }?;
        
        Ok(Models {
            model,
            tokenizer,
            device,
        })
    }
    
    fn load_model_inner(
        &self,
        config: &ConfigWrapper,
        weight_files: &[PathBuf],
    ) -> Result<(ModelWrapper, Device)> {
        // Check file extension to determine format
        if let Some(first_file) = weight_files.first() {
            let ext = first_file.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
                
            match ext {
                "gguf" => self.load_gguf_model_files(config, weight_files),
                "safetensors" => self.load_safetensors_model_files(config, weight_files),
                _ => Err(anyhow!("Unsupported file format: {}", ext))
            }
        } else {
            Err(anyhow!("No weight files provided"))
        }
    }
}

impl LLMLoaderImpl {
    /// Load GGUF model directly without external config (config embedded in GGUF)
    fn load_gguf_model_direct(
        &self,
        gguf_files: &[PathBuf],
    ) -> Result<(ModelWrapper, Device)> {
        let device = Self::device(self.use_cpu)?;
        
        let gguf_path = &gguf_files[0];
        tracing::info!("Loading GGUF model directly from: {}", gguf_path.display());
        
        let mut file = std::fs::File::open(gguf_path)?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow!("GGUF load failed: {}: {}", gguf_path.display(), e))?;
            
        // For GGUF models, we only support Qwen3 currently
        // The config is embedded in the GGUF file, so we don't need external config.json
        match self.model_type {
            ModelType::Qwen3 => {
                tracing::info!("Initializing Qwen3 quantized model from GGUF");
                let model = quantized_qwen3::ModelWeights::from_gguf(
                    content, &mut file, &device
                )?;
                
                let wrapper = LLMModelWrapper::QQwen3(model);
                Ok((ModelWrapper::LLM(wrapper), device))
            },
            _ => Err(anyhow!(
                "GGUF format currently only supported for Qwen3, got: {:?}",
                self.model_type
            ))
        }
    }

    /// Load SafeTensors files with automatic index.json/single file detection
    fn load_safetensors_files_auto(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
    ) -> Result<Vec<PathBuf>> {
        // Try to load from index.json first (for multi-file models)
        match self.load_safetensors_from_index(repo, "model.safetensors.index.json") {
            Ok(files) => {
                tracing::info!("Successfully loaded SafeTensors from index.json");
                Ok(files)
            }
            Err(index_err) => {
                tracing::info!("Index.json not found or invalid, trying single SafeTensors file: {}", index_err);
                // Fallback to single model.safetensors file
                match repo.get("model.safetensors") {
                    Ok(single_file) => {
                        tracing::info!("Successfully loaded single SafeTensors file");
                        Ok(vec![single_file])
                    }
                    Err(single_err) => {
                        Err(anyhow!(
                            "Failed to load SafeTensors: neither index.json ({}) nor single file ({}) found", 
                            index_err, single_err
                        ))
                    }
                }
            }
        }
    }

    /// Load SafeTensors from index.json (like candle_examples::hub_load_safetensors)
    fn load_safetensors_from_index(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        index_filename: &str,
    ) -> Result<Vec<PathBuf>> {
        use std::collections::HashSet;
        
        tracing::info!("Loading SafeTensors index from: {}", index_filename);
        let index_file = repo.get(index_filename)?;
        let json_data = std::fs::read_to_string(&index_file)?;
        let json: serde_json::Value = serde_json::from_str(&json_data)?;
        
        let weight_map = json
            .get("weight_map")
            .ok_or_else(|| anyhow!("Missing weight_map in index file"))?
            .as_object()
            .ok_or_else(|| anyhow!("weight_map is not an object"))?;
        
        let mut filenames = HashSet::new();
        for filename in weight_map.values() {
            if let Some(filename_str) = filename.as_str() {
                filenames.insert(filename_str);
            }
        }
        
        let mut safetensor_files = Vec::new();
        for filename in filenames {
            tracing::debug!("Fetching SafeTensor file: {}", filename);
            let file_path = repo.get(filename)?;
            safetensor_files.push(file_path);
        }
        
        tracing::info!("Loaded {} SafeTensor files from index", safetensor_files.len());
        Ok(safetensor_files)
    }

    /// Load SafeTensors format models
    fn load_safetensors_model_files(
        &self,
        config: &ConfigWrapper,
        safetensors_files: &[PathBuf],
    ) -> Result<(ModelWrapper, Device)> {
        let device = Self::device(self.use_cpu)?;
        
        // Determine DType based on device for optimization
        let dtype = if device.is_cuda() || device.is_metal() {
            DType::BF16  // GPU - fast and memory efficient
        } else {
            DType::F32   // CPU - compatibility focused
        };
        
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(safetensors_files, dtype, &device)?
        };
        
        match config {
            ConfigWrapper::LLM(llm_config) => {
                match llm_config {
                    LLMConfigWrapper::Qwen3(qwen3_config) => {
                        tracing::info!("Loading Qwen3 SafeTensors model with LM head");
                        let model = qwen3::ModelForCausalLM::new(qwen3_config, vb)?;
                        let wrapper = LLMModelWrapper::Qwen3(model);
                        Ok((ModelWrapper::LLM(wrapper), device))
                    },
                    LLMConfigWrapper::Falcon(falcon_config) => {
                        use candle_transformers::models::falcon;
                        tracing::info!("Loading Falcon SafeTensors model");
                        let model = falcon::Falcon::load(vb, falcon_config.clone())?;
                        let wrapper = LLMModelWrapper::Falcon(model);
                        Ok((ModelWrapper::LLM(wrapper), device))
                    },
                    LLMConfigWrapper::Llama(llama_config) => {
                        use candle_transformers::models::llama;
                        tracing::info!("Loading Llama SafeTensors model");
                        let model = llama::Llama::load(vb, llama_config)?;
                        let cache = llama::Cache::new(true, dtype, llama_config, &device)?;
                        let wrapper = LLMModelWrapper::Llama(model, cache);
                        Ok((ModelWrapper::LLM(wrapper), device))
                    },
                    LLMConfigWrapper::Mistral(mistral_config) => {
                        use candle_transformers::models::mistral;
                        tracing::info!("Loading Mistral SafeTensors model");
                        let model = mistral::Model::new(mistral_config, vb)?;
                        let wrapper = LLMModelWrapper::Mistral(model);
                        Ok((ModelWrapper::LLM(wrapper), device))
                    },
                    LLMConfigWrapper::StableLM(stable_config) => {
                        use candle_transformers::models::stable_lm;
                        tracing::info!("Loading StableLM SafeTensors model");
                        let model = stable_lm::Model::new(stable_config, vb)?;
                        let wrapper = LLMModelWrapper::StableLM(model);
                        Ok((ModelWrapper::LLM(wrapper), device))
                    },
                }
            },
            _ => Err(anyhow!("SafeTensors only supported for LLM models"))
        }
    }
    
    /// Load GGUF format models
    fn load_gguf_model_files(
        &self,
        config: &ConfigWrapper,
        gguf_files: &[PathBuf],
    ) -> Result<(ModelWrapper, Device)> {
        let device = Self::device(false)?; // GPU preferred for GGUF
        
        let gguf_path = &gguf_files[0];
        tracing::info!("Loading GGUF model from: {}", gguf_path.display());
        
        let mut file = std::fs::File::open(gguf_path)?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow!("GGUF load failed: {}: {}", gguf_path.display(), e))?;
            
        match config {
            ConfigWrapper::LLM(llm_config) => {
                match llm_config {
                    LLMConfigWrapper::Qwen3(_) => {
                        tracing::info!("Initializing Qwen3 quantized model");
                        let model = quantized_qwen3::ModelWeights::from_gguf(
                            content, &mut file, &device
                        )?;
                        
                        let wrapper = LLMModelWrapper::QQwen3(model);
                        Ok((ModelWrapper::LLM(wrapper), device))
                    },
                    _ => Err(anyhow!(
                        "GGUF format currently only supported for Qwen3"
                    ))
                }
            },
            _ => Err(anyhow!(
                "GGUF format only supported for LLM models"
            ))
        }
    }
}