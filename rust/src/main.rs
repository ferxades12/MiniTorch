use core::fmt;
use std::fmt::format;

use pyo3::prelude::*;
use numpy::{self, IxDyn};
use ndarray::{ArrayD};

/* /// A Python module implemented in Rust.
#[pymodule]
mod rust {
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
} */
enum Device{
    CPU,
    CUDA
}

struct Tensor{
    data: ArrayD<i32>,
    grad : Option<ArrayD<i32>>,
    grad_fn : Option<bool>,
    is_leaf : bool,
    requires_grad:bool,
    device : Device
}

// Un tensor se crea mediante un Vec lineal junto con su shape, para que los datos se almacenen contiguos en el heap
impl Tensor{
    fn new(data:Vec<i32>, shape:Vec<usize>, requires_grad:bool, device:Device) -> Tensor{
        let array: ArrayD<i32> = ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap();
        let grad: Option<ArrayD<i32>> = if requires_grad{
            Some(ArrayD::zeros(shape))
        }else {None};


        Tensor {
            data : array,
            grad : grad,
            grad_fn : None,
            is_leaf : true,
            requires_grad : requires_grad,
            device : device
        }
    }
}

impl fmt::Display for Tensor{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}" ,self.data)
    }
}

fn main(){
    let t = Tensor::new(vec![1, 2, 3, 4], vec![2, 2],  true, Device::CPU);

    print!("{}", t);
}




