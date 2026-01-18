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
    let extra_dims = grad.shape().len() - target_shape.len();
    let reduced_grad = (0..extra_dims).fold(grad.into_owned(), |acc, _|{
        acc.sum_axis(Axis(0))
    });

    let result = target_shape.iter().enumerate().fold(reduced_grad, |acc, (i, &dim)|{
        if dim == 1{
            //sum() con keepdims = True
            acc.sum_axis(Axis(i)).insert_axis(Axis(i))
        }else {
            acc
        }
    });
    
    CowArray::from(result)
}