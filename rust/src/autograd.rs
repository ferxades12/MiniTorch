use crate::tensor::{Device, Tensor};
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};

// Define tanto el enum como el metodo apply
macro_rules! define_backward_nodes {
    ($($name:ident),* $(,)?) => {
        pub enum BackwardNode{
            $($name($name),)*
        }

        impl BackwardNode{
            pub fn apply(&self, grad_output: ArrayViewD<f32>){
                match self {
                    $(BackwardNode::$name(method) => method.apply(grad_output),)*
                }
            }
        }
    };
}

define_backward_nodes!(AddBackward, MulBackward, SumBackward);

pub trait Backward {
    fn apply(&self, grad_output: ArrayViewD<f32>);
}

pub struct AddBackward {
    pub tensor: Tensor,
    pub other: Tensor,
}

impl Backward for AddBackward {
    fn apply(&self, grad_output: ArrayViewD<f32>) {
        _update_grad(&self.tensor, grad_output.clone());
        _update_grad(&self.other, grad_output);
    }
}

pub struct MulBackward {
    pub tensor: Tensor,
    pub other: Tensor,
}

impl Backward for MulBackward {
    fn apply(&self, grad_output: ArrayViewD<f32>) {
        let grad_a = match &*self.other.data {
            Device::CPU(data_b) => &grad_output * data_b,
            _ => panic!("Device not implemented"),
        };

        let grad_b = match &*self.tensor.data {
            Device::CPU(data_a) => data_a * &grad_output,
            _ => {
                panic!("Device not supported")
            }
        };

        _update_grad(&self.tensor, grad_a.view());
        _update_grad(&self.other, grad_b.view());
    }
}

pub struct SumBackward {
    pub tensor: Tensor,
    pub axis: Option<Axis>,
}

impl Backward for SumBackward {
    fn apply(&self, grad_output: ArrayViewD<f32>) {
        // insert_axis devuelve un ArrayViewD (temporal) que se destruye al final del match. Por eso se asigna el resultado a una variable previamente
        let expanded;

        let grad = match self.axis {
            None => grad_output
                .broadcast(IxDyn(&self.tensor.shape))
                .expect("Error en sum() total"),
            Some(ax) => {
                expanded = grad_output.insert_axis(ax);
                expanded
                    .broadcast(IxDyn(&self.tensor.shape))
                    .expect("Error en sum() en cierto eje")
            }
        };

        _update_grad(&self.tensor, grad);
    }
}

fn _update_grad(tensor: &Tensor, grad: ArrayViewD<f32>) {
    if !tensor.requires_grad {
        return;
    }

    if tensor.shape != grad.shape() {
        //TODO unbroadcast
    }

    if tensor.is_leaf {
        // Tomamos referencia mutable (gracias a Mutex)
        let mut grad_lock = tensor.grad.lock().unwrap();

        // Pasamos de ref mut Option a Option de ref mut para no transferir ownership de dentro a fuera del ref mut
        match grad_lock.as_mut() {
            // la impl del trait addasign esta definido como *a += &view, y no toma ownership de view
            Some(tensor_grad) => {
                *tensor_grad += &grad;
            }
            None => {
                *grad_lock = Some(grad.to_owned());
            }
        }
    } else {
        //tensor.grad_fn.backward(grad);

        tensor._backward(Some(grad));
    }
}
