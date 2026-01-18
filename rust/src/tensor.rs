use core::fmt;
use std::sync::{Arc, Mutex};

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Axis, IxDyn};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyNotImplementedError, prelude::*};

use crate::autograd::{BackwardNode, SumBackward};
use crate::cpu;
use crate::util;

type Grad = Arc<Mutex<Option<ArrayD<f32>>>>;

pub fn new_grad(grad: Option<ArrayD<f32>>) -> Grad {
    Arc::new(Mutex::new(grad))
}

#[pyclass] // para exponer el struct a Python
pub struct Tensor {
    pub data: Arc<Device>,
    //Mutex para tener varias referencias mutables. Arc para pasar varias referencias de forma eficiente
    pub grad: Grad,
    pub grad_fn: Option<Arc<BackwardNode>>,
    pub shape: Vec<usize>, // Para futura implementacion de CUDA
    #[pyo3(get)] // Permite leer is_leaf desde Python
    pub is_leaf: bool,
    #[pyo3(get)] // Permite leer requires_grad desde Python
    pub requires_grad: bool,
    //TODO Device
}

pub enum Device {
    CPU(ArrayD<f32>),
    CUDA(bool),
}
impl fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CPU(_) => write!(f, "CPU"),
            Device::CUDA(_) => write!(f, "CUDA"),
        }
    }
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
impl Tensor {
    #[new] // #[new] marca el constructor para Python
    #[pyo3(signature = (data, requires_grad=false))] // Para asignar atributos opcionales
    fn new(data: PyReadonlyArrayDyn<f32>, requires_grad: bool) -> PyResult<Tensor> {
        /*  Al usar to_owned(), evitas errores de segmentación (segfaults) que ocurrirían si Python
            decidiera liberar la memoria del array original mientras Rust todavía está realizando cálculos sobre él
            Tambien lo transforma  a un ArrayD
        */
        let array: ArrayD<f32> = data.as_array().to_owned();
        let shape: Vec<usize> = array.shape().to_vec();

        let grad: Option<ArrayD<f32>> =
            if requires_grad { Some(ArrayD::zeros(IxDyn(&shape))) } else { None };

        Ok(Tensor {
            data: Arc::new(Device::CPU(array)),
            grad: new_grad(grad),
            grad_fn: None,
            shape: shape,
            is_leaf: true,
            requires_grad: requires_grad,
        })
    }

    // Método para convertir de vuelta a numpy
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        match &*self.data {
            Device::CPU(array) => Ok(PyArrayDyn::from_array(py, &array)),
            _ => Err(PyNotImplementedError::new_err("Not implemented")),
        }
    }

    // Getter para grad que retorna numpy array
    #[getter]
    fn grad<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        let grad_lock = self.grad.lock().unwrap();
        match &*grad_lock {
            Some(grad_array) => Ok(Some(PyArrayDyn::from_array(py, grad_array))),
            None => Ok(None),
        }
    }

    // Python operator overloads
    fn __add__(&self, rhs: &Tensor) -> PyResult<Tensor> {
        Ok(self + rhs)
    }

    fn __mul__(&self, rhs: &Tensor) -> PyResult<Tensor> {
        Ok(self * rhs)
    }

    fn __sub__(&self, rhs: &Tensor) -> PyResult<Tensor> {
        Ok(self - rhs)
    }

    fn __truediv__(&self, rhs: &Tensor) -> PyResult<Tensor> {
        Ok(self / rhs)
    }

    fn abs(&self) -> PyResult<Tensor> {
        Ok(self.clone()._abs())
    }

    fn t(&self) -> PyResult<Tensor> {
        Ok(self.clone()._transpose())
    }

    fn __pow__(&self, rhs: &Tensor, _modulo: Option<&Tensor>) -> PyResult<Tensor> {
        Ok(self.clone()._pow(rhs))
    }
    #[pyo3(signature = (axis=None))]
    fn sum(&self, axis: Option<usize>) -> PyResult<Tensor> {
        /* let ax = match axis {
            None => None,
            Some(ax) => Some(Axis(ax))
        }; */

        let ax = axis.map(Axis);
        let result = self.dispatch_unary_op_with_axes(ax, cpu::sum_cpu, |tensor: Tensor| {
            BackwardNode::SumBackward(SumBackward { tensor: tensor, axis: ax })
        });

        Ok(result)
    }

    #[pyo3(signature = (grad=None))]
    pub fn backward(&self, grad: Option<PyReadonlyArrayDyn<f32>>) -> PyResult<()> {
        let grad_view = grad.as_ref().map(|g| g.as_array());
        self._backward(grad_view);
        Ok(())
    }

    fn __repr__(&self) -> String {
        match &*self.data {
            Device::CPU(array) => {
                format!("Tensor(shape={:?}, requires_grad={})", array.shape(), self.requires_grad)
            }
            _ => format!("CUDA not implemented"),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl Tensor {
    fn dispatch_scalar_op<F>(
        &self,
        rhs: f32,
        reverse: bool,
        kernel_cpu: fn(ArrayViewD<f32>, ArrayViewD<f32>, ArrayViewMutD<f32>),
        make_node: F,
    ) -> Tensor
    where
        F: FnOnce(Tensor, Tensor) -> BackwardNode,
    {
        let rhs_tensor = match *self.data{
            Device::CPU(_) => result_tensor_no_requires_grad(Device::CPU(ArrayD::from_elem(IxDyn(&[]), rhs))).unwrap(),
            Device::CUDA(_) => panic!("CUDA not implemented")
        };

        if reverse {
            rhs_tensor.dispatch_binary_op(self, kernel_cpu, make_node)
        } else {
            self.dispatch_binary_op(&rhs_tensor, kernel_cpu, make_node)
        }
    }

    fn dispatch_binary_op<F>(
        &self,
        rhs: &Tensor,
        kernel_cpu: fn(ArrayViewD<f32>, ArrayViewD<f32>, ArrayViewMutD<f32>),
        make_node: F,
    ) -> Tensor
    where
        F: FnOnce(Tensor, Tensor) -> BackwardNode,
    {
        let out = match (&*self.data, &*rhs.data) {
            (Device::CPU(a), Device::CPU(b)) => {
                // Broadcasting
                let shape;
                let (data_a, data_b) = if a.shape() != b.shape() {
                    shape = util::broadcast_shapes(a.shape(), b.shape()).expect("Las formas de los tensores no son compatibles para broadcasting");

                    (a.broadcast(shape.clone()).unwrap(), b.broadcast(shape.clone()).unwrap())
                } else {
                    shape = a.shape().to_vec();
                    (a.view(), b.view())
                };

                let mut out = ArrayD::zeros(IxDyn(&shape));
                kernel_cpu(data_a, data_b, out.view_mut());
                Device::CPU(out)
            }
            _ => {
                panic!("Unimplemented");
            }
        };

        if self.requires_grad || rhs.requires_grad {
            let grad_fn = Some(Arc::new(make_node(self.clone(), rhs.clone())));

            result_tensor_requires_grad(out, None, grad_fn).unwrap()
        } else {
            result_tensor_no_requires_grad(out).unwrap()
        }
    }

    // El shape de out es fijo
    fn dispatch_unary_op<F>(
        &self,
        kernel_cpu: fn(ArrayViewD<f32>, ArrayViewMutD<f32>),
        make_node: F,
    ) -> Tensor
    where
        F: FnOnce(Tensor) -> BackwardNode,
    {
        let out = match &*self.data {
            Device::CPU(a) => {
                let mut out = ArrayD::zeros(IxDyn(&self.shape));
                kernel_cpu(a.view(), out.view_mut());
                Device::CPU(out)
            }
            _ => {
                panic!("Unimplemented");
            }
        };

        if self.requires_grad {
            let grad_fn = Some(Arc::new(make_node(self.clone())));

            result_tensor_requires_grad(out, None, grad_fn).unwrap()
        } else {
            result_tensor_no_requires_grad(out).unwrap()
        }
    }

    fn dispatch_unary_op_with_axes<F>(
        &self,
        axis: Option<Axis>,
        kernel_cpu: fn(ArrayViewD<f32>, Option<Axis>, ArrayViewMutD<f32>),
        make_node: F,
    ) -> Tensor
    where
        F: FnOnce(Tensor) -> BackwardNode,
    {
        let out = match &*self.data {
            Device::CPU(a) => {
                let out_shape = match axis {
                    None => vec![],
                    Some(ax) => self
                        .shape
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != ax.index())
                        .map(|(_, &dim)| dim)
                        .collect(),
                };

                let mut out = ArrayD::zeros(IxDyn(&out_shape));
                kernel_cpu(a.view(), axis, out.view_mut());
                Device::CPU(out)
            }
            _ => {
                panic!("Unimplemented");
            }
        };

        if self.requires_grad {
            let grad_fn = Some(Arc::new(make_node(self.clone()))); //TODO mirar si es realmente necesario el make_node en caso unario

            result_tensor_requires_grad(out, None, grad_fn).unwrap()
        } else {
            result_tensor_no_requires_grad(out).unwrap()
        }
    }

    fn _pow(self, rhs: &Tensor) -> Tensor {
        self.dispatch_binary_op(rhs, cpu::pow_cpu, |a: Tensor, b: Tensor| {
            BackwardNode::PowBackward(crate::autograd::PowBackward { tensor: a, other: b })
        })
    }

    fn numel(&self) -> usize {
        self.shape.iter().fold(1, |acc, &x| acc * x)
    }

    pub fn _backward(&self, grad: Option<ArrayViewD<f32>>) {
        // Validar que grad es Some para tensores no-escalares
        if self.numel() != 1 && grad.is_none() {
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
            Some(grad_fn) => grad_fn.apply(grad_view),
        }
    }
}

fn result_tensor_requires_grad(
    data: Device,
    grad: Option<ArrayD<f32>>,
    grad_fn: Option<Arc<BackwardNode>>,
) -> Result<Tensor, String> {
    let shape = match &data {
        Device::CPU(array) => array.shape().to_vec(),
        _ => {
            return Err("Not implemented".into());
        }
    };

    let grad = grad.or_else(|| None);

    Ok(Tensor {
        data: Arc::new(data),
        grad: new_grad(grad),
        grad_fn: grad_fn,
        shape: shape,
        is_leaf: false,
        requires_grad: true,
    })
}

fn result_tensor_no_requires_grad(data: Device) -> Result<Tensor, String> {
    let shape = match &data {
        Device::CPU(array) => array.shape().to_vec(),
        _ => {
            return Err("Not implemented".into());
        }
    };

    Ok(Tensor {
        data: Arc::new(data),
        grad: new_grad(None),
        grad_fn: None,
        shape: shape,
        is_leaf: true,
        requires_grad: false,
    })
}

macro_rules!  impl_scalar_op{
    ($($trait:ident, $t:ty);*) => {
        $(
            paste::paste! {
                // &Tensor op scalar
                impl std::ops::$trait<$t> for &Tensor {
                    type Output = Tensor;
                    fn [<$trait:lower>](self, rhs: $t) -> Self::Output {
                        self.dispatch_scalar_op(
                            rhs as f32,
                            false,
                            cpu::[<$trait:lower _cpu>],
                            |a: Tensor, b: Tensor| BackwardNode::[<$trait Backward>](
                                crate::autograd::[<$trait Backward>] { tensor: a, other: b }
                            )
                        )
                    }
                }

                //Tensor op scalar
                impl std::ops::$trait<$t> for Tensor {
                    type Output = Tensor;
                    fn [<$trait:lower>](self, rhs: $t) -> Self::Output {
                        (&self).[<$trait:lower>](rhs as f32)
                    }
                }

                //scalar op &Tensor
                impl std::ops::$trait<&Tensor> for $t {
                    type Output = Tensor;
                    fn [<$trait:lower>](self, rhs: &Tensor) -> Self::Output {
                        rhs.dispatch_scalar_op(
                            self as f32,
                            true,
                            cpu::[<$trait:lower _cpu>],
                            |a: Tensor, b: Tensor| BackwardNode::[<$trait Backward>](
                                crate::autograd::[<$trait Backward>] { tensor: a, other: b }
                            )
                        )
                    }
                }

                //scalar op Tensor
                impl std::ops::$trait<Tensor> for $t {
                    type Output = Tensor;
                    fn [<$trait:lower>](self, rhs: Tensor) -> Self::Output {
                        rhs.[<$trait:lower>](self as f32)
                    }
                }
            }
        )*
    };
}


macro_rules! impl_tensor_op {
    ($($trait:ident);*) => {
        $( //Multiples llamadas
            paste::paste! { // Para pasar Add en vez de (Add, add, AddBackward)
                // 1. &Tensor op &Tensor
                impl std::ops::$trait<&Tensor> for &Tensor {
                    type Output = Tensor;
                    fn [<$trait:lower>](self, rhs: &Tensor) -> Self::Output {
                        self.dispatch_binary_op(
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

macro_rules! impl_binary_op {
    ($($trait:ident);*) => {
        $(
            impl_tensor_op!($trait);
            impl_scalar_op!($trait, f32);
            impl_scalar_op!($trait, i32);
            impl_scalar_op!($trait, i64);
            impl_scalar_op!($trait, u32);
            impl_scalar_op!($trait, u64);
            impl_scalar_op!($trait, f64);
        )*
    };
}

macro_rules! impl_unary_op {
    ($($trait:ident);*) => {
        $( //Multiples llamadas
            paste::paste! { // Para pasar Add en vez de (Add, add, AddBackward)
                impl Tensor{
                    pub fn [<_ $trait:lower>](self) -> Tensor {
                        self.dispatch_unary_op(
                            cpu::[<$trait:lower _cpu>],
                            |a: Tensor| BackwardNode::[<$trait Backward>](
                                crate::autograd::[<$trait Backward>] { tensor: a }
                            )
                        )
                    }
                }
            }
        )*
    };
}

impl_binary_op!(Add; Mul; Sub; Div);

impl_unary_op!(Abs; Transpose);