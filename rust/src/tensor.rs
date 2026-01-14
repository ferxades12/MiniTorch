use std::sync::{Arc, Mutex};

use pyo3::{exceptions::PyNotImplementedError, prelude::*};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use ndarray::{Array0, Array1, ArrayD, ArrayViewD, ArrayViewMutD, Axis, IxDyn};

use crate::cpu;
use crate::autograd::{BackwardNode, SumBackward};

type Grad = Arc<Mutex<Option<ArrayD<f32>>>>;

pub fn new_grad(grad: Option<ArrayD<f32>>)-> Grad {
    Arc::new(Mutex::new(grad))
}


#[pyclass] // para exponer el struct a Python
pub struct Tensor{
    pub data: Arc<Device>,  
    //Mutex para tener varias referencias mutables. Arc para pasar varias referencias de forma eficiente
    pub grad : Grad,
    pub grad_fn : Option<Arc<BackwardNode>>,
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
            grad_fn: self.grad_fn.as_ref().map(|arc| Arc::clone(arc)),
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

    // Getter para grad que retorna numpy array
    #[getter]
    fn grad<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        let grad_lock = self.grad.lock().unwrap();
        match &*grad_lock {
            Some(grad_array) => Ok(Some(PyArrayDyn::from_array(py, grad_array))),
            None => Ok(None)
        }
    }

    // Python operator overloads
    fn __add__(&self, rhs: &Tensor) -> PyResult<Tensor> {
        Ok(self + rhs)
    }

    fn __mul__(&self, rhs: &Tensor) -> PyResult<Tensor> {
        Ok(self * rhs)
    }

    #[pyo3(signature = (axis=None))]
    fn sum(&self, axis:Option<usize>) -> PyResult<Tensor>{
        /* let ax = match axis {
            None => None,
            Some(ax) => Some(Axis(ax))
        }; */

        let ax = axis.map(Axis);
        let result = self.dispatch_unary_op_with_axes(ax, cpu::sum_cpu, |tensor: Tensor|{
            BackwardNode::SumBackward(SumBackward { 
                tensor: tensor, 
                axis: ax 
            })
        });

        Ok(result)
    }

    #[pyo3(signature = (grad=None))]
    pub fn backward(&self, grad:Option<PyReadonlyArrayDyn<f32>>) -> PyResult<()>{
        let grad_view = grad.as_ref().map(|g| g.as_array());
        self._backward(grad_view);
        Ok(())
    }
    
}

impl Tensor{
    fn result_tensor_requires_grad(&self, data:Device, grad:Option<ArrayD<f32>>, grad_fn:Option<Arc<BackwardNode>>) -> Result<Tensor, String> {
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
            let grad_fn = Some(Arc::new(make_node(self.clone(), rhs.clone())));

            self.result_tensor_requires_grad(out, None, grad_fn).unwrap()
        } else {
            self.result_tensor_no_requires_grad(out).unwrap()
        }
    }

    // El shape de out es fijo
    fn dispatch_unary_op<F>(&self, kernel_cpu: fn(&ArrayViewD<f32>, &mut ArrayViewMutD<f32>), make_node:F) -> Tensor 
    where
        F:FnOnce(Tensor) -> BackwardNode
    {
        let out = match &*self.data {
            Device::CPU(a) => {
                let mut out = ArrayD::zeros(IxDyn(&self.shape));
                kernel_cpu(&a.view(), &mut out.view_mut());
                Device::CPU(out)
            }
            _ => {panic!("Unimplemented");}
        };

       
        if self.requires_grad {
            let grad_fn = Some(Arc::new(make_node(self.clone())));

            self.result_tensor_requires_grad(out, None, grad_fn).unwrap()
        } else {
            self.result_tensor_no_requires_grad(out).unwrap()
        }
    }

    // El shape de out es fijo (Escalar)
    fn dispatch_unary_op_with_axes<F>(&self, axis: Option<Axis>, kernel_cpu: fn(&ArrayViewD<f32>, Option<Axis>,&mut ArrayViewMutD<f32>), make_node:F) -> Tensor 
    where
        F:FnOnce(Tensor) -> BackwardNode
    {
        let out = match &*self.data {
            Device::CPU(a) => {
                let mut out = ArrayD::zeros(IxDyn(&[])); //IxDyn Ligeramente ineficiente pero no quiero tirarme 6h en mejorarlo
                kernel_cpu(&a.view(), axis, &mut out.view_mut());
                Device::CPU(out)
            }
            _ => {panic!("Unimplemented");}
        };

       
        if self.requires_grad {
            let grad_fn = Some(Arc::new(make_node(self.clone()))); //TODO mirar si es realmente necesario el make_node en caso unario

            self.result_tensor_requires_grad(out, None, grad_fn).unwrap()
        } else {
            self.result_tensor_no_requires_grad(out).unwrap()
        }
    }

    fn numel(&self) -> usize{
        self.shape.iter().fold(1, |acc, &x| acc * x)
    }

    pub fn _backward(&self, grad: Option<ArrayViewD<f32>>){
        // Validar que grad es Some para tensores no-escalares
        if self.numel() != 1 && grad.is_none(){
            panic!("grad can be implicitly created only for scalars");
        }

        // Usar el grad proporcionado o crear uno de unos
        let grad_array;
        let grad_view = match grad {
            Some(g) => g,
            None => {
                grad_array = ArrayD::ones(IxDyn(&self.shape));
                grad_array.view()
            }
        };

        match self.grad_fn.as_ref() {
            None => panic!("Tensor has no gradient function"),
            Some(grad_fn) => grad_fn.apply(grad_view)
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