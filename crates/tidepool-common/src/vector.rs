use std::f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

impl DistanceMetric {
    pub fn parse(metric: Option<&str>) -> Self {
        match metric.unwrap_or("") {
            "cosine" | "cosine_distance" => Self::Cosine,
            "euclidean" | "euclidean_squared" => Self::Euclidean,
            "dot" | "dot_product" => Self::DotProduct,
            _ => Self::Cosine,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cosine => "cosine_distance",
            Self::Euclidean => "euclidean_squared",
            Self::DotProduct => "dot_product",
        }
    }
}

pub fn distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Euclidean => euclidean_squared(a, b),
        DistanceMetric::DotProduct => -dot_product(a, b),
    }
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 2.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 2.0;
    }

    let similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
    (1.0 - similarity) as f32
}

pub fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }

    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = *x as f64 - *y as f64;
        sum += diff * diff;
    }

    sum as f32
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MIN;
    }

    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += *x as f64 * *y as f64;
    }

    sum as f32
}

pub fn normalize(v: &[f32]) -> Vec<f32> {
    let mut sum = 0.0f64;
    for val in v {
        let f = *val as f64;
        sum += f * f;
    }

    if sum == 0.0 {
        return v.to_vec();
    }

    let norm = sum.sqrt() as f32;
    v.iter().map(|val| *val / norm).collect()
}

pub fn magnitude(v: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    for val in v {
        sum += (*val as f64) * (*val as f64);
    }
    sum.sqrt() as f32
}
