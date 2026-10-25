use canary_rs::{ExecutionConfig, SessionConfig};

pub fn execution_config_from_env() -> ExecutionConfig {
    let mut config = ExecutionConfig::new();

    #[cfg(feature = "coreml")]
    {
        let compute_units = std::env::var("CANARY_COREML_COMPUTE_UNITS")
            .unwrap_or_default()
            .to_lowercase();
        let compute_units = match compute_units.trim() {
            "all" => canary_rs::CoreMLComputeUnits::All,
            "cpu_and_gpu" => canary_rs::CoreMLComputeUnits::CPUAndGPU,
            "cpu_only" => canary_rs::CoreMLComputeUnits::CPUOnly,
            _ => canary_rs::CoreMLComputeUnits::CPUAndNeuralEngine,
        };
        config = config.with_coreml_compute_units(compute_units);

        let low_precision = std::env::var("CANARY_COREML_LOW_PRECISION")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        config = config.with_coreml_low_precision(low_precision);
    }

    #[cfg(feature = "cuda")]
    {
        let device_id = std::env::var("CANARY_CUDA_DEVICE_ID")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(0);
        config = config.with_cuda_device_id(device_id);
    }

    let itn = std::env::var("CANARY_ITN").ok().and_then(|value| {
        match value.trim().to_lowercase().as_str() {
            "1" | "true" => Some(true),
            "0" | "false" => Some(false),
            _ => None,
        }
    });
    config = config.with_session_config(SessionConfig::new().with_itn(itn));

    config
}
