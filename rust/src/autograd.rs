use crate::tensor::{Device, Tensor};
use ndarray::{ArrayD, ArrayViewD, Axis, CowArray, IxDyn};

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

define_backward_nodes!(AddBackward, MulBackward, SubBackward, DivBackward, SumBackward);

pub trait Backward {
    fn apply(&self, grad_output: ArrayViewD<f32>);
}

macro_rules! implement_bitwise_backward {
    /*
       $name : Nombre del nodo a implementar (AddBackward)
       $funca : funcion que calcula el gradiente de tensor
       $funcb : funcion que calcula el gradiente de other
       El resultado debe ser ArrayD o ArrayViewD, que posteriormente se pasara a CowArray en la macro
    */
    ($name:ident, $funca:expr , $funcb:expr ) => {
        pub struct $name {
            pub tensor: Tensor,
            pub other: Tensor,
        }

        impl Backward for $name {
            fn apply(&self, grad_output: ArrayViewD<f32>) {
                let (grad_a, grad_b) = match (&*self.tensor.data, &*self.other.data) {
                    (Device::CPU(data_a), Device::CPU(data_b)) => (
                        CowArray::from($funca(data_a, data_b, grad_output.clone())),
                        CowArray::from($funcb(data_a, data_b, grad_output)),
                    ),
                    _ => {
                        panic!(
                            "Combinación de dispositivos no soportada: ({:?}, {:?})",
                            &*self.tensor.data, &*self.other.data
                        );
                    }
                };

                _update_grad(&self.tensor, grad_a);
                _update_grad(&self.other, grad_b);
            }
        }
    };
}

implement_bitwise_backward!(
    AddBackward,
    |_data_a, _data_b, grad_output| grad_output,
    |_data_a, _data_b, grad_output| grad_output
);

implement_bitwise_backward!(
    MulBackward,
    |_data_a, data_b, grad_output| &grad_output * data_b,
    |data_a, _data_b, grad_output| data_a * &grad_output
);

implement_bitwise_backward!(
    SubBackward,
    |_data_a, _data_b, grad_output| grad_output,
    |_data_a, _data_b, grad_output: ArrayViewD<f32>| -&grad_output
);

implement_bitwise_backward!(
    DivBackward,
    |_data_a, data_b, grad_output| &grad_output / data_b,
    |data_a: &ArrayD<f32>, data_b: &ArrayD<f32>, grad_output| {
        (-data_a / data_b.mapv(|a| a * a)) * grad_output
    }
);

pub struct SumBackward {
    pub tensor: Tensor,
    pub axis: Option<Axis>,
}

impl Backward for SumBackward {
    fn apply(&self, grad_output: ArrayViewD<f32>) {
        // insert_axis devuelve un ArrayViewD (temporal) que se destruye al final del match. Por eso se asigna el resultado a una variable previamente
        let expanded;

        let grad = match self.axis {
            None => grad_output.broadcast(IxDyn(&self.tensor.shape)).expect("Error en sum() total"),
            Some(ax) => {
                expanded = grad_output.insert_axis(ax);
                expanded
                    .broadcast(IxDyn(&self.tensor.shape))
                    .expect("Error en sum() en cierto eje")
            }
        };

        _update_grad(&self.tensor, CowArray::from(grad));
    }
}

fn _update_grad(tensor: &Tensor, grad: CowArray<f32, IxDyn>) {
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
                *tensor_grad += &grad.view();
            }
            None => {
                *grad_lock = Some(grad.into_owned());
            }
        }
    } else {
        //tensor.grad_fn.backward(grad);

        tensor._backward(Some(grad.view()));
    }
}
