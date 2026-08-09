//! API key newtype with redacted formatting.

/// An API key: redacted `Debug`/`Display`, no `Serialize`, explicit
/// [`ApiKey::expose`] accessor. No memory zeroing — wrap your own type if
/// you need that (§ 10).
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    /// Wraps a key.
    #[must_use]
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    /// Returns the secret value. The only way to read the key.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl From<String> for ApiKey {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ApiKey {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey([REDACTED])")
    }
}

impl std::fmt::Display for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[REDACTED]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacted_formatting() {
        let key = ApiKey::new("sk-secret");
        assert_eq!(format!("{key:?}"), "ApiKey([REDACTED])");
        assert_eq!(format!("{key}"), "[REDACTED]");
        assert_eq!(key.expose(), "sk-secret");
    }
}
