// Package config handles environment variable configuration for Tidepool services.
package config

import (
	"os"
	"time"
)

// Config holds all configuration values for Tidepool services.
type Config struct {
	// S3 configuration
	AWSAccessKeyID     string
	AWSSecretAccessKey string
	AWSEndpointURL     string
	AWSRegion          string
	BucketName         string

	// Service configuration
	CacheDir           string
	Namespace          string
	CompactionInterval time.Duration
}

// Load reads configuration from environment variables.
func Load() *Config {
	cfg := &Config{
		AWSAccessKeyID:     os.Getenv("AWS_ACCESS_KEY_ID"),
		AWSSecretAccessKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
		AWSEndpointURL:     os.Getenv("AWS_ENDPOINT_URL"),
		AWSRegion:          os.Getenv("AWS_REGION"),
		BucketName:         os.Getenv("BUCKET_NAME"),
		CacheDir:           getEnvOrDefault("CACHE_DIR", "/data"),
		Namespace:          getEnvOrDefault("NAMESPACE", "default"),
		CompactionInterval: parseDuration(os.Getenv("COMPACTION_INTERVAL"), 5*time.Minute),
	}
	return cfg
}

// Validate checks that required configuration values are present.
func (c *Config) Validate() error {
	if c.AWSAccessKeyID == "" {
		return &ConfigError{Field: "AWS_ACCESS_KEY_ID"}
	}
	if c.AWSSecretAccessKey == "" {
		return &ConfigError{Field: "AWS_SECRET_ACCESS_KEY"}
	}
	if c.AWSEndpointURL == "" {
		return &ConfigError{Field: "AWS_ENDPOINT_URL"}
	}
	if c.AWSRegion == "" {
		return &ConfigError{Field: "AWS_REGION"}
	}
	if c.BucketName == "" {
		return &ConfigError{Field: "BUCKET_NAME"}
	}
	return nil
}

// ConfigError represents a missing configuration value.
type ConfigError struct {
	Field string
}

func (e *ConfigError) Error() string {
	return "missing required configuration: " + e.Field
}

func getEnvOrDefault(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}

func parseDuration(s string, defaultValue time.Duration) time.Duration {
	if s == "" {
		return defaultValue
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return defaultValue
	}
	return d
}
