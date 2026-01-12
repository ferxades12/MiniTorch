use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, IxDyn, azip};

pub fn add_cpu(a: &ArrayViewD<f32>, b: &ArrayViewD<f32>, out: &mut ArrayViewMutD<f32>){
    azip!((out in out, &a in a, &b in b) *out = a + b);
}

fn mul_cpu(a: &ArrayViewD<f32>, b: &ArrayViewD<f32>, out: &mut ArrayViewMutD<f32>){
    azip!((out in out, &a in a, &b in b) *out = a * b);
}

fn sub_cpu(a: &ArrayViewD<f32>, b: &ArrayViewD<f32>, out: &mut ArrayViewMutD<f32>){
    azip!((out in out, &a in a, &b in b) *out = a - b);
}

fn pow_cpu(a: &ArrayViewD<f32>, b: &ArrayViewD<f32>, out: &mut ArrayViewMutD<f32>){
    azip!((out in out, &a in a, &b in b) *out = a.powf(b));
}

/* fn dot_cpu(a: &ArrayViewD<f32>, b: &ArrayViewD<f32>, out: &mut ArrayViewMutD<f32>){
    ndarray::linalg::general_mat_mul(1.0, a, b, 0.0, &mut out);
} */