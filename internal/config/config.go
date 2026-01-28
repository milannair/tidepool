// Package config handles environment variable configuration for Tidepool services.
package config

import (
	"os"
	"strconv"
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
	Port               string
	ReadTimeout        time.Duration
	WriteTimeout       time.Duration
	IdleTimeout        time.Duration
	MaxBodyBytes       int64
	MaxTopK            int
	CORSAllowOrigin    string
}

// Load reads configuration from environment variables.
// Supports both Railway bucket names and standard AWS names.
func Load() *Config {
	cfg := &Config{
		AWSAccessKeyID:     os.Getenv("AWS_ACCESS_KEY_ID"),
		AWSSecretAccessKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
		AWSEndpointURL:     os.Getenv("AWS_ENDPOINT_URL"),
		AWSRegion:          getEnvWithFallback("AWS_DEFAULT_REGION", "AWS_REGION"),
		BucketName:         getEnvWithFallback("AWS_S3_BUCKET_NAME", "BUCKET_NAME"),
		CacheDir:           getEnvOrDefault("CACHE_DIR", "/data"),
		Namespace:          getEnvOrDefault("NAMESPACE", "default"),
		CompactionInterval: parseDuration(os.Getenv("COMPACTION_INTERVAL"), 5*time.Minute),
		Port:               getEnvOrDefault("PORT", "8080"),
		ReadTimeout:        parseDuration(os.Getenv("READ_TIMEOUT"), 30*time.Second),
		WriteTimeout:       parseDuration(os.Getenv("WRITE_TIMEOUT"), 60*time.Second),
		IdleTimeout:        parseDuration(os.Getenv("IDLE_TIMEOUT"), 60*time.Second),
		MaxBodyBytes:       parseInt64(os.Getenv("MAX_BODY_BYTES"), 25*1024*1024),
		MaxTopK:            parseInt(os.Getenv("MAX_TOP_K"), 1000),
		CORSAllowOrigin:    getEnvOrDefault("CORS_ALLOW_ORIGIN", "*"),
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
		return &ConfigError{Field: "AWS_DEFAULT_REGION or AWS_REGION"}
	}
	if c.BucketName == "" {
		return &ConfigError{Field: "AWS_S3_BUCKET_NAME or BUCKET_NAME"}
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

// getEnvWithFallback tries primary key first, then fallback key.
func getEnvWithFallback(primary, fallback string) string {
	if v := os.Getenv(primary); v != "" {
		return v
	}
	return os.Getenv(fallback)
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

func parseInt64(s string, defaultValue int64) int64 {
	if s == "" {
		return defaultValue
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return defaultValue
	}
	return v
}

func parseInt(s string, defaultValue int) int {
	if s == "" {
		return defaultValue
	}
	v, err := strconv.Atoi(s)
	if err != nil {
		return defaultValue
	}
	return v
}
