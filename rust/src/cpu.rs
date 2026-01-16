use ndarray::{arr0, azip, ArrayD, ArrayViewD, ArrayViewMutD, Axis, IxDyn};

pub fn add_cpu(a: &ArrayViewD<f32>, b: &ArrayViewD<f32>, out: &mut ArrayViewMutD<f32>) {
    azip!((out in out, &a in a, &b in b) *out = a + b);
}

pub fn mul_cpu(a: &ArrayViewD<f32>, b: &ArrayViewD<f32>, out: &mut ArrayViewMutD<f32>) {
    azip!((out in out, &a in a, &b in b) *out = a * b);
}

pub fn sub_cpu(a: &ArrayViewD<f32>, b: &ArrayViewD<f32>, out: &mut ArrayViewMutD<f32>) {
    azip!((out in out, &a in a, &b in b) *out = a - b);
}

pub fn pow_cpu(a: &ArrayViewD<f32>, b: &ArrayViewD<f32>, out: &mut ArrayViewMutD<f32>) {
    azip!((out in out, &a in a, &b in b) *out = a.powf(b));
}

pub fn sum_cpu(a: &ArrayViewD<f32>, axis: Option<Axis>, out: &mut ArrayViewMutD<f32>) {
    match axis {
        None => {
            out.fill(a.sum());
        }
        Some(ax) => {
            out.assign(&a.sum_axis(ax));
        }
    }
}
