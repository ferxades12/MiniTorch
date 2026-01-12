use std::fmt::format;

use pyo3::{exceptions::PyNotImplementedError, prelude::*};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use ndarray::{ArrayD, IxDyn};

#[pyclass] // para exponer el struct a Python
pub struct Tensor{
    data: Device,  
    grad : Option<ArrayD<f32>>,
    grad_fn : Option<bool>,
    shape: Vec<usize>, // Para futura implementacion de CUDA
    #[pyo3(get)]  // Permite leer is_leaf desde Python
    pub is_leaf : bool,
    #[pyo3(get)]  // Permite leer requires_grad desde Python
    pub requires_grad: bool,
    //TODO Device
}

enum Device{
    CPU(ArrayD<f32>),
    CUDA(bool)
}

// Métodos que Python puede llamar
#[pymethods]
impl Tensor{
    #[new] // #[new] marca el constructor para Python
    #[pyo3(signature = (data, requires_grad=false))]  // Para asignar atributos opcionales
    fn new(data: PyReadonlyArrayDyn<f32>, requires_grad: bool) -> PyResult<Tensor> {
        /*  Al usar to_owned(), evitas errores de segmentación (segfaults) que ocurrirían si Python 
            decidiera liberar la memoria del array original mientras Rust todavía está realizando cálculos sobre él 
            Tambien lo transforma  a un ArrayD
        */
        let array:ArrayD<f32>  = data.as_array().to_owned();
        let shape: Vec<usize> = array.shape().to_vec();
        
        let grad: Option<ArrayD<f32>> = if requires_grad{
            Some(ArrayD::zeros(IxDyn(&shape)))
        } else {None};

        Ok(Tensor {
            data : Device::CPU(array),
            grad : grad,
            grad_fn : None,
            shape: shape,
            is_leaf : true,
            requires_grad : requires_grad,
        })
    }
    
    // Método para convertir de vuelta a numpy
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        match &self.data {
            Device::CPU(array) =>{Ok(PyArrayDyn::from_array(py, &array))}
            _ => Err(PyNotImplementedError::new_err("Not implemented"))
        }
    }
    
    // Exponer __repr__ para Python
    fn __repr__(&self) -> String {
        match &self.data {
            Device::CPU(array) => format!("Tensor(shape={:?}, requires_grad={})", array.shape(), self.requires_grad),
            _ => format!("CUDA not implemented")
        }
    }
}
