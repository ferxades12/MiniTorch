mod autograd;
mod cpu;
mod tensor;

use pyo3::prelude::*;
use tensor::Tensor;

/// Módulo Python rustorch
#[pymodule]
fn rustorch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?; // Registrar la clase Tensor
    Ok(())
}
