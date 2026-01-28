use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A JSON-like attribute value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttrValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<AttrValue>),
    Object(BTreeMap<String, AttrValue>),
}

/// Attributes stored as a map of string keys to values.
/// Serialized as JSON for storage compatibility.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Attributes(pub BTreeMap<String, AttrValue>);

impl Attributes {
    pub fn get(&self, key: &str) -> Option<&AttrValue> {
        self.0.get(key)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::ops::Deref for Attributes {
    type Target = BTreeMap<String, AttrValue>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Attributes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
