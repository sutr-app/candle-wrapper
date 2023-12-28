use anyhow::{anyhow, Result};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_transformers::models::bert::{self, BertModel};
use hf_hub::api::sync::ApiRepo;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::Deserialize;
use serde::Serialize;
use std::path::PathBuf;
use tokenizers::Tokenizer;

// typical weight and setting filenames
pub const WEIGHT_FILES_SAFETENSOR_2FILES: [&'static str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];
pub const WEIGHT_FILES_SAFETENSOR_SINGLE: [&str; 1] = ["model.safetensors"];
pub const WEIGHT_FILES_GGUF_SINGLE: [&str; 1] = ["model-q4k.gguf"]; // not used

pub const TOKENIZER_FILE: &'static str = "tokenizer.json";
pub const MODEL_CONFIG_FILE: &'static str = "config.json";

pub const MAIN_REVISION: &'static str = "main";

pub enum ConfigWrapper {
    Bert(bert::Config),
}

pub enum ModelWrapper {
    Bert(BertModel, bert::Config),
}

impl ModelWrapper {
    // TODO implement for LLM
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        _seqlen_offset: usize,
    ) -> Result<Tensor> {
        match (self, token_type_ids) {
            (Self::Bert(m, _), Some(ttids)) => m.forward(input_ids, ttids).map_err(|e| e.into()),
            (Self::Bert(_, _), None) => Err(anyhow!("token type ids is necessary for Bert model")),
        }
    }
}

// TODO change file types (pth, safetensor, onnx): only safetensor
// const definitions for each model are only availables
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HFModelSetting {
    pub model_id: String,
    pub revision: Option<String>,       // default: main
    pub tokenizer_file: Option<String>, // default: tokenizer.json
    pub weight_files: Option<Vec<String>>,
    pub model_config_file: Option<String>, // default: config.json
}

impl HFModelSetting {
    pub fn from_model_id(model_id: String) -> Self {
        Self {
            model_id,
            revision: None,
            tokenizer_file: None,
            weight_files: None,
            model_config_file: None,
        }
    }
}

pub struct Models {
    pub model: ModelWrapper,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

// TODO using type matching
pub trait HFModelLoader {
    // GPU or CPU (GPU is only available for cuda id:0 device)
    fn device(cpu: bool) -> Result<Device> {
        if cpu {
            Ok(Device::Cpu)
        } else if cuda_is_available() {
            Ok(Device::new_cuda(0)?)
        } else if metal_is_available() {
            Ok(Device::new_metal(0)?)
        } else {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                println!(
                    "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
                );
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                println!(
                    "Running on CPU, to run on GPU, build this example with `--features cuda`"
                );
            }
            Ok(Device::Cpu)
        }
    }
    // TODO resolve from model_config
    fn dtype(&self, device: &Device) -> DType;
    /// load config using inner load()
    fn load_config_inner(&self, repo: &ApiRepo, config_file: &PathBuf) -> Result<ConfigWrapper>;

    /// load model using inner load()
    // TODO select model from config
    // TODO use proper load method by weight files extension
    fn load_model_inner(
        &self,
        config: &ConfigWrapper,
        weight_files: &Vec<PathBuf>,
    ) -> Result<(ModelWrapper, Device)>;

    // devide each model to each files
    fn load_from_setting(&self, setting: HFModelSetting) -> Result<Models> {
        let api = Api::new()?;
        tracing::info!("load hf model: {}", &setting.model_id);
        let repo = api.repo(Repo::with_revision(
            setting.model_id.clone(),
            RepoType::Model,
            setting
                .revision
                .as_ref()
                .unwrap_or(&MAIN_REVISION.to_string())
                .clone(), // XXX default revision
        ));
        let tokenizer_filename = match setting.tokenizer_file.as_ref() {
            Some(file) => std::path::PathBuf::from(file),
            None => repo.get(&TOKENIZER_FILE)?,
        };
        // TODO error handling
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
            .flat_map(|f| repo.get(f))
            .collect::<Vec<_>>();
        tracing::debug!("load tokenizer: {:?}", &tokenizer_filename);
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("error in load tokenizer: {:?}", e))?;

        let mfile = std::path::PathBuf::from(
            setting
                .model_config_file
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or(MODEL_CONFIG_FILE),
        );

        let config = self.load_config_inner(&repo, &mfile)?;
        tracing::debug!("load model: {:?}", &filenames);
        let (model, device) = self.load_model_inner(&config, &filenames)?;

        tracing::info!("got model: {}", &setting.model_id);
        Ok(Models {
            model,
            tokenizer,
            device,
        })
    }
}
