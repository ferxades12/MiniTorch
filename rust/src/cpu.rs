use ndarray::{azip, ArrayViewD, ArrayViewMutD, Axis};
use numpy::Ix2;

pub fn add_cpu(a: ArrayViewD<f32>, b: ArrayViewD<f32>, out: ArrayViewMutD<f32>) {
    azip!((out in out, &a in a, &b in b) *out = a + b);
}

pub fn mul_cpu(a: ArrayViewD<f32>, b: ArrayViewD<f32>, out: ArrayViewMutD<f32>) {
    azip!((out in out, &a in a, &b in b) *out = a * b);
}

pub fn sub_cpu(a: ArrayViewD<f32>, b: ArrayViewD<f32>, out: ArrayViewMutD<f32>) {
    azip!((out in out, &a in a, &b in b) *out = a - b);
}

pub fn pow_cpu(a: ArrayViewD<f32>, b: ArrayViewD<f32>, out: ArrayViewMutD<f32>) {
    azip!((out in out, &a in a, &b in b) *out = a.powf(b));
}

pub fn dot_cpu(a: ArrayViewD<f32>, b: ArrayViewD<f32>, out: ArrayViewMutD<f32>) {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let k_dim_a = a_shape[a_shape.len() - 1];
    let k_dim_b = b_shape[0];

    let rows_a = a_shape.iter().take(a_shape.len() - 1).product();
    let cols_b = b_shape.iter().skip(1).product();

    let a_2d = a.to_shape(Ix2(rows_a, k_dim_a)).unwrap();
    let b_2d = b.to_shape(Ix2(k_dim_b, cols_b)).unwrap();

    /* out.to_shape(Ix2(rows_a, cols_b)); //to_shape no modifica el objeto
    ndarray::linalg::general_mat_mul(1.0, &a_2d, &b_2d, 0.0, out); */

    /* Poco eficiente:
    let mut result_2d = ndarray::Array2::zeros((rows_a, cols_b));

    // Debido a que out es un ArrayViewMutD, cualquier cambio realizado a través de la vista out_2d afecta directamente al mismo bloque de memoria
    ndarray::linalg::general_mat_mul(1.0, &a_2d, &b_2d, 0.0, &mut result_2d);


    let result_shape = out.shape().to_vec();
    let result_reshaped = result_2d.to_shape(ndarray::IxDyn(&result_shape)).unwrap();
    out.assign(&result_reshaped); */

    // into_shape_with_order ya que general_mat_mul requiere que los datos esten de forma contigua en memoria
    let mut out_2d = out.into_shape_with_order((rows_a, cols_b)).unwrap();
    ndarray::linalg::general_mat_mul(1.0, &a_2d, &b_2d, 0.0, &mut out_2d);

}

pub fn div_cpu(a: ArrayViewD<f32>, b: ArrayViewD<f32>, out: ArrayViewMutD<f32>) {
    azip!((out in out, &a in a, &b in b) *out = a / b);
}

pub fn sum_cpu(a: ArrayViewD<f32>, axis: Option<Axis>, mut out: ArrayViewMutD<f32>) {
    match axis {
        None => {
            out.fill(a.sum());
        }
        Some(ax) => {
            out.assign(&a.sum_axis(ax));
        }
    }
}

pub fn abs_cpu(a: ArrayViewD<f32>, out: ArrayViewMutD<f32>) {
    azip!((out in out, &a in a) *out = a.abs());
}

pub fn transpose_cpu(a: ArrayViewD<f32>, mut out: ArrayViewMutD<f32>) {
    out.assign(&a.t());
}

pub fn log_cpu(a: ArrayViewD<f32>, out: ArrayViewMutD<f32>) {
    azip!((out in out, &a in a) *out = a.ln())
}