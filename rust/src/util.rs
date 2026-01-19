use ndarray::{ArrayD, Axis, CowArray, IxDyn};

pub fn broadcast_shapes(a:&[usize], b:&[usize]) -> Option<Vec<usize>>{
    let mut output = Vec::new();
    let mut i1 = a.iter().rev();
    let mut i2 = b.iter().rev();

    loop {
        match (i1.next(), i2.next()) {
            (Some(&n1), Some(&n2)) =>{
                if n1 == n2 {output.push(n1);}
                else if n1 == 1 {
                    output.push(n2);
                }
                else if n2 == 1 {
                    output.push(n1);
                }
                else {
                    return None;
                }
            },
            (Some(&n1), None) | (None,  Some(&n1)) =>{output.push(n1);},
            (None, None) =>{break;}
        };
    }
    output.reverse();
    Some(output)
}


pub fn get_dot_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    match (a.len(), b.len()) {
        (1, 1) => {
            assert!(a[0] == b[0]);
            vec![]
        }
        (2, 1) => {
            assert!(a[1] == b[0]);
            vec![a[0]]
        }
        (1, 2) => {
            assert!(a[0] == b[0]);
            vec![b[1]]
        }
        (2, 2) => {
            assert!(a[1] == b[0]);
            vec![a[0], b[1]]
        }
        _ => panic!("Dot product only supports 1D or 2D tensors."),
    }
}


pub fn unbroadcast(grad:CowArray<f32, IxDyn>, target_shape:Vec<usize>) -> CowArray<f32, IxDyn>{
    if grad.shape() == target_shape {
        return grad;
    }

    let mut current_grad = grad.into_owned();

    let extra_dims = current_grad.shape().len().saturating_sub(target_shape.len());
    for _ in 0..extra_dims {
        current_grad = current_grad.sum_axis(Axis(0));
    }

    for (i, &target_dim) in target_shape.iter().enumerate() {
        if target_dim == 1 && current_grad.shape()[i] > 1 {
            current_grad = current_grad
                .sum_axis(Axis(i))
                .insert_axis(Axis(i));
        }
    }

    CowArray::from(current_grad)
}