// Package storage provides S3-compatible object storage operations.
package storage

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"path"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	tidepoolconfig "github.com/tidepool/tidepool/internal/config"
)

// Client wraps S3 operations for Tidepool.
type Client struct {
	s3     *s3.Client
	bucket string
}

// NewClient creates a new S3 storage client.
func NewClient(ctx context.Context, cfg *tidepoolconfig.Config) (*Client, error) {
	awsCfg, err := config.LoadDefaultConfig(ctx,
		config.WithRegion(cfg.AWSRegion),
		config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(
			cfg.AWSAccessKeyID,
			cfg.AWSSecretAccessKey,
			"",
		)),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS config: %w", err)
	}

	s3Client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		o.BaseEndpoint = aws.String(cfg.AWSEndpointURL)
		o.UsePathStyle = true
	})

	return &Client{
		s3:     s3Client,
		bucket: cfg.BucketName,
	}, nil
}

// Get retrieves an object from S3.
func (c *Client) Get(ctx context.Context, key string) ([]byte, error) {
	resp, err := c.s3.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(c.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get object %s: %w", key, err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read object %s: %w", key, err)
	}
	return data, nil
}

// Put stores an object in S3.
func (c *Client) Put(ctx context.Context, key string, data []byte) error {
	_, err := c.s3.PutObject(ctx, &s3.PutObjectInput{
		Bucket: aws.String(c.bucket),
		Key:    aws.String(key),
		Body:   bytes.NewReader(data),
	})
	if err != nil {
		return fmt.Errorf("failed to put object %s: %w", key, err)
	}
	return nil
}

// Append appends data to an existing object or creates a new one.
// Note: S3 doesn't support true append, so this reads, appends, and writes.
func (c *Client) Append(ctx context.Context, key string, data []byte) error {
	existing, err := c.Get(ctx, key)
	if err != nil {
		// Object doesn't exist, create new
		return c.Put(ctx, key, data)
	}
	// Append to existing
	combined := append(existing, data...)
	return c.Put(ctx, key, combined)
}

// Delete removes an object from S3.
func (c *Client) Delete(ctx context.Context, key string) error {
	_, err := c.s3.DeleteObject(ctx, &s3.DeleteObjectInput{
		Bucket: aws.String(c.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return fmt.Errorf("failed to delete object %s: %w", key, err)
	}
	return nil
}

// List returns all keys with the given prefix.
func (c *Client) List(ctx context.Context, prefix string) ([]string, error) {
	var keys []string
	paginator := s3.NewListObjectsV2Paginator(c.s3, &s3.ListObjectsV2Input{
		Bucket: aws.String(c.bucket),
		Prefix: aws.String(prefix),
	})

	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to list objects with prefix %s: %w", prefix, err)
		}
		for _, obj := range page.Contents {
			keys = append(keys, *obj.Key)
		}
	}
	return keys, nil
}

// Exists checks if an object exists in S3.
func (c *Client) Exists(ctx context.Context, key string) (bool, error) {
	_, err := c.s3.HeadObject(ctx, &s3.HeadObjectInput{
		Bucket: aws.String(c.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return false, nil
	}
	return true, nil
}

// NamespacePath returns the full S3 path for a namespace-relative path.
func NamespacePath(namespace, subpath string) string {
	return path.Join("namespaces", namespace, subpath)
}

// ManifestPath returns the path to the manifest file.
func ManifestPath(namespace, version string) string {
	return NamespacePath(namespace, path.Join("manifests", version+".json"))
}

// LatestManifestPath returns the path to the latest manifest.
func LatestManifestPath(namespace string) string {
	return ManifestPath(namespace, "latest")
}

// WALPath returns the path to a WAL file.
func WALPath(namespace, date, uuid string) string {
	return NamespacePath(namespace, path.Join("wal", date, uuid+".jsonl"))
}

// WALPrefix returns the prefix for all WAL files.
func WALPrefix(namespace string) string {
	return NamespacePath(namespace, "wal/")
}

// SegmentPath returns the path to a segment file.
func SegmentPath(namespace, segmentID string) string {
	return NamespacePath(namespace, path.Join("segments", segmentID+".parquet"))
}

// IndexPath returns the path to an index file.
func IndexPath(namespace, segmentID string) string {
	return NamespacePath(namespace, path.Join("indexes", segmentID+".idx"))
}
