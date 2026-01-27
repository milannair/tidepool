// Package vector provides vector similarity calculations.
package vector

import (
	"math"
)

// DistanceMetric defines the type of distance calculation.
type DistanceMetric string

const (
	Cosine     DistanceMetric = "cosine_distance"
	Euclidean  DistanceMetric = "euclidean_squared"
	DotProduct DistanceMetric = "dot_product"
)

// ParseMetric parses a distance metric string.
func ParseMetric(s string) DistanceMetric {
	switch s {
	case "cosine", "cosine_distance":
		return Cosine
	case "euclidean", "euclidean_squared":
		return Euclidean
	case "dot", "dot_product":
		return DotProduct
	default:
		return Cosine // Default to cosine
	}
}

// Distance calculates the distance between two vectors using the specified metric.
func Distance(a, b []float32, metric DistanceMetric) float32 {
	switch metric {
	case Cosine:
		return CosineDistance(a, b)
	case Euclidean:
		return EuclideanSquared(a, b)
	case DotProduct:
		// For dot product, higher is better, so we negate for ranking
		return -DotProductSimilarity(a, b)
	default:
		return CosineDistance(a, b)
	}
}

// CosineDistance calculates 1 - cosine similarity.
// Returns 0 for identical vectors, 2 for opposite vectors.
func CosineDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 2.0 // Maximum distance for invalid inputs
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 2.0
	}

	similarity := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	return float32(1.0 - similarity)
}

// EuclideanSquared calculates the squared Euclidean distance.
// Using squared distance avoids expensive sqrt operation.
func EuclideanSquared(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.MaxFloat32)
	}

	var sum float64
	for i := range a {
		diff := float64(a[i]) - float64(b[i])
		sum += diff * diff
	}

	return float32(sum)
}

// DotProductSimilarity calculates the dot product between two vectors.
// Higher values indicate more similar vectors.
func DotProductSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.MinInt32)
	}

	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}

	return float32(sum)
}

// Normalize normalizes a vector to unit length (L2 norm).
func Normalize(v []float32) []float32 {
	var sum float64
	for _, val := range v {
		sum += float64(val) * float64(val)
	}

	if sum == 0 {
		return v
	}

	norm := float32(math.Sqrt(sum))
	result := make([]float32, len(v))
	for i, val := range v {
		result[i] = val / norm
	}

	return result
}

// Magnitude calculates the L2 norm (magnitude) of a vector.
func Magnitude(v []float32) float32 {
	var sum float64
	for _, val := range v {
		sum += float64(val) * float64(val)
	}
	return float32(math.Sqrt(sum))
}
