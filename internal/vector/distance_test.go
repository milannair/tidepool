package vector

import "testing"

func TestDistances(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{1, 0}
	c := []float32{-1, 0}

	if got := CosineDistance(a, b); got != 0 {
		t.Fatalf("expected cosine distance 0, got %v", got)
	}
	if got := CosineDistance(a, c); got < 1.99 {
		t.Fatalf("expected cosine distance near 2, got %v", got)
	}

	if got := EuclideanSquared([]float32{1, 2}, []float32{2, 4}); got != 5 {
		t.Fatalf("expected euclidean squared 5, got %v", got)
	}

	if got := DotProductSimilarity([]float32{1, 2}, []float32{3, 4}); got != 11 {
		t.Fatalf("expected dot product 11, got %v", got)
	}

	if got := Distance(a, b, DotProduct); got != -1 {
		t.Fatalf("expected distance -1 for dot product, got %v", got)
	}
}
