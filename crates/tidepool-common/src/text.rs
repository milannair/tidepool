use std::collections::HashSet;

use rust_stemmers::{Algorithm, Stemmer};
use unicode_segmentation::UnicodeSegmentation;

pub trait Tokenizer: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<String>;
}

#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub language: String,
    pub enable_stemming: bool,
    pub stopwords: HashSet<String>,
    pub min_token_len: usize,
    pub max_token_len: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            language: "english".to_string(),
            enable_stemming: true,
            stopwords: default_stopwords(),
            min_token_len: 2,
            max_token_len: 32,
        }
    }
}

impl TokenizerConfig {
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }
}

pub struct DefaultTokenizer {
    config: TokenizerConfig,
    stemmer_alg: Option<Algorithm>,
}

impl DefaultTokenizer {
    pub fn new(config: TokenizerConfig) -> Self {
        let stemmer_alg = if config.enable_stemming {
            Some(algorithm_from_lang(&config.language))
        } else {
            None
        };
        Self { config, stemmer_alg }
    }

    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }
}

impl Tokenizer for DefaultTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let min_len = self.config.min_token_len;
        let max_len = self.config.max_token_len;
        let stopwords = &self.config.stopwords;
        let stemmer = self.stemmer_alg.map(Stemmer::create);

        for word in text.unicode_words() {
            let mut token = word.to_lowercase();
            if let Some(stemmer) = &stemmer {
                token = stemmer.stem(&token).to_string();
            }
            let len = token.len();
            if len < min_len || len > max_len {
                continue;
            }
            if stopwords.contains(&token) {
                continue;
            }
            tokens.push(token);
        }

        tokens
    }
}

fn algorithm_from_lang(lang: &str) -> Algorithm {
    match lang.to_lowercase().as_str() {
        "english" | "en" => Algorithm::English,
        "spanish" | "es" => Algorithm::Spanish,
        "french" | "fr" => Algorithm::French,
        "german" | "de" => Algorithm::German,
        "italian" | "it" => Algorithm::Italian,
        "dutch" | "nl" => Algorithm::Dutch,
        "portuguese" | "pt" => Algorithm::Portuguese,
        "russian" | "ru" => Algorithm::Russian,
        "swedish" | "sv" => Algorithm::Swedish,
        "norwegian" | "no" => Algorithm::Norwegian,
        _ => Algorithm::English,
    }
}

fn default_stopwords() -> HashSet<String> {
    let words = [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
        "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
        "there", "these", "they", "this", "to", "was", "will", "with", "you", "your",
    ];
    words.iter().map(|w| (*w).to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokenizer = DefaultTokenizer::new(TokenizerConfig::default());
        let tokens = tokenizer.tokenize("The quick brown fox jumps");
        assert!(tokens.contains(&"quick".to_string()));
        assert!(!tokens.contains(&"the".to_string()));
    }

    #[test]
    fn test_tokenize_no_stemming() {
        let cfg = TokenizerConfig {
            enable_stemming: false,
            ..TokenizerConfig::default()
        };
        let tokenizer = DefaultTokenizer::new(cfg);
        let tokens = tokenizer.tokenize("running runner");
        assert!(tokens.contains(&"running".to_string()));
    }
}
