use crate::model::ExecutionConfig;
use crate::types::Result;
use ort::ep::ExecutionProviderDispatch;
use ort::session::builder::SessionBuilder;

pub fn apply_execution_providers(
    mut builder: SessionBuilder,
    config: &ExecutionConfig,
    use_cpu: bool,
) -> Result<SessionBuilder> {
    #[cfg(not(any(feature = "coreml", feature = "cuda")))]
    let _ = config;
    let providers = if use_cpu {
        vec![cpu_provider()]
    } else {
        #[cfg(feature = "coreml")]
        {
            vec![coreml_provider(config)?, cpu_provider()]
        }
        #[cfg(all(not(feature = "coreml"), feature = "cuda"))]
        {
            vec![cuda_provider(config.cuda_device_id), cpu_provider()]
        }
        #[cfg(all(not(feature = "coreml"), not(feature = "cuda"), feature = "webgpu"))]
        {
            vec![webgpu_provider(), cpu_provider()]
        }
        #[cfg(not(any(feature = "coreml", feature = "cuda", feature = "webgpu")))]
        {
            vec![cpu_provider()]
        }
    };
    builder = builder.with_execution_providers(providers)?;
    Ok(builder)
}

fn cpu_provider() -> ExecutionProviderDispatch {
    ort::ep::CPU::default().build()
}

#[cfg(feature = "cuda")]
fn cuda_provider(device_id: i32) -> ExecutionProviderDispatch {
    ort::ep::CUDA::default().with_device_id(device_id).build()
}

#[cfg(feature = "webgpu")]
fn webgpu_provider() -> ExecutionProviderDispatch {
    ort::ep::WebGPU::default().build()
}

#[cfg(feature = "coreml")]
fn coreml_provider(config: &ExecutionConfig) -> Result<ExecutionProviderDispatch> {
    let mut provider = ort::ep::CoreML::default()
        .with_compute_units(
            config
                .coreml_compute_units
                .unwrap_or(ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine),
        )
        .with_low_precision_accumulation_on_gpu(config.coreml_low_precision);
    if let Some(model_format) = config.coreml_model_format {
        provider = provider.with_model_format(model_format);
    }
    if config.coreml_static_input_shapes {
        provider = provider.with_static_input_shapes(true);
    }
    if config.coreml_enable_subgraphs {
        provider = provider.with_subgraphs(true);
    }
    if let Some(strategy) = config.coreml_specialization_strategy {
        provider = provider.with_specialization_strategy(strategy);
    }
    if let Some(cache_dir) = config.coreml_cache_dir.as_deref() {
        if !cache_dir.trim().is_empty() {
            provider = provider.with_model_cache_dir(cache_dir.to_string());
        }
    }
    Ok(provider.build())
}
