mod tensor;
mod cpu;

use pyo3::prelude::*;
use tensor::Tensor;

/// Módulo Python rs_torch
#[pymodule]
fn rs_torch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;  // Registrar la clase Tensor
    Ok(())
}