use tidepool_common::quantization::{
    f16_distance,
    load_binary,
    marshal_binary,
    quantize_f16,
    quantize_sq8,
    sq8_distance,
    QuantizationKind,
    Sq8Query,
};
use tidepool_common::simd;
use tidepool_common::vector::{distance_with_norms, DistanceMetric};

#[test]
fn f16_roundtrip_and_distance() {
    let vectors: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let q = quantize_f16(&refs).expect("quantize f16");
    let data = marshal_binary(&q).expect("marshal f16");
    let loaded = load_binary(&data).expect("load f16");

    assert_eq!(loaded.kind, QuantizationKind::F16);
    assert_eq!(loaded.vector_count, 2);
    assert_eq!(loaded.dimensions, 2);

    let query = vec![1.0, 2.0];
    let query_norm = simd::l2_norm_f32(&query);
    let v0_norm = simd::l2_norm_f32(&vectors[0]);
    let approx = f16_distance(
        &query,
        loaded.vector_bytes(0).unwrap(),
        DistanceMetric::Cosine,
        query_norm,
        v0_norm,
    );
    let exact = distance_with_norms(&query, &vectors[0], query_norm, v0_norm, DistanceMetric::Cosine);
    assert!(
        (approx - exact).abs() < 0.05,
        "approx {} exact {}",
        approx,
        exact
    );
}

#[test]
fn sq8_roundtrip_and_ordering() {
    let vectors: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let q = quantize_sq8(&refs).expect("quantize sq8");
    let data = marshal_binary(&q).expect("marshal sq8");
    let loaded = load_binary(&data).expect("load sq8");

    assert_eq!(loaded.kind, QuantizationKind::SQ8);
    assert_eq!(loaded.vector_count, 2);
    assert_eq!(loaded.dimensions, 2);

    let query = vec![1.0, 0.0];
    let query_norm = simd::l2_norm_f32(&query);
    let v0_norm = simd::l2_norm_f32(&vectors[0]);
    let v1_norm = simd::l2_norm_f32(&vectors[1]);

    let sq8_query = Sq8Query::new(&query, &loaded.scales, &loaded.mins, query_norm);
    let d0 = sq8_distance(
        &sq8_query,
        loaded.vector_bytes(0).unwrap(),
        &loaded.scales,
        &loaded.mins,
        DistanceMetric::Cosine,
        v0_norm,
    );
    let d1 = sq8_distance(
        &sq8_query,
        loaded.vector_bytes(1).unwrap(),
        &loaded.scales,
        &loaded.mins,
        DistanceMetric::Cosine,
        v1_norm,
    );
    assert!(d0 < d1, "expected first vector closer: d0={} d1={}", d0, d1);
}
