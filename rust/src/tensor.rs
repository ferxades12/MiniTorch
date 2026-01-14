use std::sync::{Arc, Mutex};

use pyo3::{exceptions::PyNotImplementedError, prelude::*};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use ndarray::{ArrayD, IxDyn, ArrayViewD, ArrayViewMutD};

use crate::cpu;
use crate::autograd::BackwardNode;

type Grad = Arc<Mutex<Option<ArrayD<f32>>>>;

pub fn new_grad(grad: Option<ArrayD<f32>>)-> Grad {
    Arc::new(Mutex::new(grad))
}


#[pyclass] // para exponer el struct a Python
pub struct Tensor{
    pub data: Arc<Device>,  
    //Mutex para tener varias referencias mutables. Arc para pasar varias referencias de forma eficiente
    pub grad : Grad,
    pub grad_fn : Option<Box<BackwardNode>>,
    pub shape: Vec<usize>, // Para futura implementacion de CUDA
    #[pyo3(get)]  // Permite leer is_leaf desde Python
    pub is_leaf : bool,
    #[pyo3(get)]  // Permite leer requires_grad desde Python
    pub requires_grad: bool,
    //TODO Device
}

pub enum Device{
    CPU(ArrayD<f32>),
    CUDA(bool)
}
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            data: Arc::clone(&self.data),
            grad: Arc::clone(&self.grad),
            grad_fn: None, // No clonamos el grafo computacional
            shape: self.shape.clone(),
            is_leaf: self.is_leaf,
            requires_grad: self.requires_grad,
        }
    }
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
            data : Arc::new(Device::CPU(array)),
            grad : new_grad(grad),
            grad_fn : None,
            shape: shape,
            is_leaf : true,
            requires_grad : requires_grad,
        })
    }

     // Método para convertir de vuelta a numpy
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        match &*self.data {
            Device::CPU(array) =>{Ok(PyArrayDyn::from_array(py, &array))}
            _ => Err(PyNotImplementedError::new_err("Not implemented"))
        }
    }

    // Python operator overloads
    fn __add__(&self, rhs: &Tensor) -> PyResult<Tensor> {
        Ok(self + rhs)
    }

    fn __mul__(&self, rhs: &Tensor) -> PyResult<Tensor> {
        Ok(self * rhs)
    }

    #[pyo3(signature = (grad=None))]
    fn backward(&self, grad:Option<PyReadonlyArrayDyn<f32>>) -> PyResult<()>{
        let grad_arc = match grad {
            Some(g) => new_grad(Some(g.as_array().to_owned())),
            None => new_grad(None)
        };

        self._backward(grad_arc);
        Ok(())
    }
    
}

impl Tensor{
    fn result_tensor_requires_grad(&self, data:Device, grad:Option<ArrayD<f32>>, grad_fn:Option<Box<BackwardNode>>) -> Result<Tensor, String> {
        let shape = match &data {
            Device::CPU(array) => array.shape().to_vec(),
            _ => {return Err("Not implemented".into());}
        };

        let grad = grad.or_else(|| None);


        Ok(Tensor {
            data : Arc::new(data),
            grad : new_grad(grad),
            grad_fn : grad_fn,
            shape: shape,
            is_leaf : false,
            requires_grad : true,
        })
    }

    fn result_tensor_no_requires_grad(&self, data:Device) -> Result<Tensor, String> {
        let shape = match &data {
            Device::CPU(array) => array.shape().to_vec(),
            _ => {return Err("Not implemented".into());}
        };

        Ok(Tensor {
            data : Arc::new(data),
            grad : new_grad(None),
            grad_fn : None,
            shape: shape,
            is_leaf : true,
            requires_grad : false,
        })
    }

    // Exponer __repr__ para Python
    fn __repr__(&self) -> String {
        match &*self.data {
            Device::CPU(array) => format!("Tensor(shape={:?}, requires_grad={})", array.shape(), self.requires_grad),
            _ => format!("CUDA not implemented")
        }
    }


    fn dispatch_binary_op<F>(&self, method:&str, rhs:&Tensor, kernel_cpu: fn(&ArrayViewD<f32>, &ArrayViewD<f32>, &mut ArrayViewMutD<f32>), make_node:F) -> Tensor 
    where
        F:FnOnce(Tensor, Tensor) -> BackwardNode
    {
        if self.shape != rhs.shape{
            panic!("Shapes dont match for op {}", method);
        }
        
        let out = match (&*self.data, &*rhs.data) {
            (Device::CPU(a), Device::CPU(b)) => {
                let mut out = ArrayD::zeros(IxDyn(&self.shape));
                kernel_cpu(&a.view(), &b.view(), &mut out.view_mut());
                Device::CPU(out)
            }
            _ => {panic!("Unimplemented");}
        };

       
        if self.requires_grad || rhs.requires_grad {
            let grad_fn = Some(Box::new(make_node(self.clone(), rhs.clone())));

            self.result_tensor_requires_grad(out, None, grad_fn).unwrap()
        } else {
            self.result_tensor_no_requires_grad(out).unwrap()
        }
    }

    fn numel(&self) -> usize{
        self.shape.iter().fold(1, |acc, &x| acc * x)
    }

    fn  _backward(&self, grad: Grad){
        let mut grad_lock = grad.lock().unwrap();
        let grad_owned = grad_lock.take(); // Mueve el valor sin clonar

        if self.numel() != 1 && grad_owned.is_none(){
            panic!("'grad can be implicitly created only for scalars'");
        }

        let grad = match grad_owned {
            Some(gradient) => new_grad(Some(gradient)),
            None => new_grad(Some(ArrayD::ones(IxDyn(&self.shape))))
        };

        match self.grad_fn.as_ref() {
            None => panic!("Tensor has no gradient function"),
            // Arc<Mutex<Option<ArrayD>>> -> MutexGuard<Option<ArrayD>> -> &ArrayD -> ArrayViewD
            Some(grad_fn) => grad_fn.apply(grad.lock().unwrap().as_ref().unwrap().view())
        }
    }
}


macro_rules! impl_binary_op {
    ($($trait:ident);*) => {
        $( //Multiples llamadas
            paste::paste! { // Para pasar Add en vez de (Add, add, AddBackward)
                // 1. &Tensor op &Tensor
                impl std::ops::$trait<&Tensor> for &Tensor {
                    type Output = Tensor;
                    fn [<$trait:lower>](self, rhs: &Tensor) -> Self::Output {
                        self.dispatch_binary_op(
                            stringify!([<$trait:lower>]), 
                            rhs, 
                            cpu::[<$trait:lower _cpu>],
                            |a: Tensor, b: Tensor| BackwardNode::[<$trait Backward>](
                                crate::autograd::[<$trait Backward>] { tensor: a, other: b }
                            )
                        )
                    }
                }

                // 2. Tensor op Tensor
                impl std::ops::$trait<Tensor> for Tensor {
                    type Output = Tensor;
                    fn [<$trait:lower>](self, rhs: Tensor) -> Self::Output {
                        (&self).[<$trait:lower>](&rhs)
                    }
                }

                // 3. Tensor op &Tensor
                impl std::ops::$trait<&Tensor> for Tensor {
                    type Output = Tensor;
                    fn [<$trait:lower>](self, rhs: &Tensor) -> Self::Output {
                        (&self).[<$trait:lower>](rhs)
                    }
                }

                // 4. &Tensor op Tensor
                impl std::ops::$trait<Tensor> for &Tensor {
                    type Output = Tensor;
                    fn [<$trait:lower>](self, rhs: Tensor) -> Self::Output {
                        self.[<$trait:lower>](&rhs)
                    }
                }
            }
        )*
    };
}


impl_binary_op!(Add; Mul);