//! The namespaced `extra` escape hatch and its RFC 7396 merge machinery.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Format-namespaced free-form JSON attached to every IR node.
///
/// Keys are canonical format ids (`openai_chat_completions`, …). When
/// serializing to format `F`, only `extra[F]` is merged into that node's JSON
/// output with JSON Merge Patch semantics (RFC 7396): objects merge
/// recursively, arrays and scalars replace, `null` **deletes** the key at any
/// depth. Parsing collects unknown fields into the source format's namespace.
///
/// Because `null` means delete, `extra` cannot set a field to literal JSON
/// `null`; use the `on_request` hook for that rare case.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Extra(BTreeMap<String, Map<String, Value>>);

impl Extra {
    /// Creates an empty `Extra`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` when no namespace holds any field.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.values().all(Map::is_empty)
    }

    /// Returns the fields stored for a format id.
    #[must_use]
    pub fn get(&self, format: &str) -> Option<&Map<String, Value>> {
        self.0.get(format)
    }

    /// Returns the (created-on-demand) mutable namespace for a format id.
    pub fn namespace_mut(&mut self, format: &str) -> &mut Map<String, Value> {
        self.0.entry(format.to_owned()).or_default()
    }

    /// Sets a top-level field in a format's namespace.
    pub fn set(&mut self, format: &str, key: impl Into<String>, value: impl Into<Value>) {
        self.namespace_mut(format).insert(key.into(), value.into());
    }

    /// Removes and returns a format's whole namespace.
    pub fn remove(&mut self, format: &str) -> Option<Map<String, Value>> {
        self.0.remove(format)
    }

    /// Iterates over the format ids that have a namespace.
    pub fn formats(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }

    /// Builds an `Extra` holding one namespace of unknown fields collected
    /// while parsing. Null-valued fields are dropped — the documented
    /// representational loss (§ 1): for an unknown field the library cannot
    /// know whether `null` carries distinct semantics, and `null` means
    /// delete in the merge.
    #[must_use]
    pub fn from_unknown(format: &str, fields: Map<String, Value>) -> Self {
        let filtered: Map<String, Value> =
            fields.into_iter().filter(|(_, v)| !v.is_null()).collect();
        let mut map = BTreeMap::new();
        if !filtered.is_empty() {
            map.insert(format.to_owned(), filtered);
        }
        Self(map)
    }

    /// Merges this node's namespace for `format` into `target` (RFC 7396),
    /// recording every replace/delete in `log` with pointers made absolute
    /// against `base` (the node's pointer in the final output JSON).
    pub fn merge_into(&self, format: &str, target: &mut Value, base: &str, log: &mut MergeLog) {
        if let Some(patch) = self.0.get(format)
            && !patch.is_empty()
        {
            merge_patch(target, patch, base, log);
        }
    }
}

/// One recorded merge operation kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MergeOpKind {
    /// A non-object value (scalar or array) was set at the pointer.
    SetScalar,
    /// An object was set where the target had no object (created or replaced
    /// a non-object) — the subtree at the pointer was rebuilt.
    SetTree,
    /// The key at the pointer was deleted (`null` in the patch).
    Deleted,
}

/// Log of the paths an `extra` merge set or deleted, used to mark warnings
/// as `overridden` (§ 6): a warning is overridden when `extra` set or deleted
/// exactly its pointer, or set a non-object value (or delete) at an ancestor
/// of it — under RFC 7396 arrays and scalars replace while objects merge, so
/// only those constitute an ancestor override.
#[derive(Clone, Debug, Default)]
pub struct MergeLog {
    ops: Vec<(MergeOpKind, String)>,
}

impl MergeLog {
    /// Creates an empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` when `extra` explicitly addressed `pointer` per the § 6 rule.
    #[must_use]
    pub fn overrides(&self, pointer: &str) -> bool {
        self.ops.iter().any(|(kind, p)| {
            p == pointer
                || (matches!(kind, MergeOpKind::SetScalar | MergeOpKind::Deleted)
                    && pointer.starts_with(p.as_str())
                    && pointer.as_bytes().get(p.len()) == Some(&b'/'))
        })
    }

    /// `true` when nothing was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// Escapes one JSON Pointer reference token (RFC 6901).
#[must_use]
pub fn escape_pointer_token(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

/// Applies an RFC 7396 merge patch of `patch` onto `target`, recording
/// operations into `log` with pointers absolute against `base`.
pub fn merge_patch(target: &mut Value, patch: &Map<String, Value>, base: &str, log: &mut MergeLog) {
    if !target.is_object() {
        *target = Value::Object(Map::new());
    }
    let obj = target.as_object_mut().expect("just ensured object");
    for (key, pv) in patch {
        let ptr = format!("{base}/{}", escape_pointer_token(key));
        match pv {
            Value::Null => {
                obj.remove(key);
                log.ops.push((MergeOpKind::Deleted, ptr));
            }
            Value::Object(po) => {
                let entry = obj.entry(key.clone()).or_insert(Value::Object(Map::new()));
                if !entry.is_object() {
                    *entry = Value::Object(Map::new());
                    log.ops.push((MergeOpKind::SetTree, ptr.clone()));
                }
                merge_patch(entry, po, &ptr, log);
            }
            other => {
                obj.insert(key.clone(), other.clone());
                log.ops.push((MergeOpKind::SetScalar, ptr));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn patch_of(v: Value) -> Map<String, Value> {
        match v {
            Value::Object(m) => m,
            _ => panic!("patch must be an object"),
        }
    }

    #[test]
    fn merge_replaces_scalars_and_arrays() {
        let mut target = json!({"a": 1, "b": [1, 2], "keep": "x"});
        let mut log = MergeLog::new();
        merge_patch(&mut target, &patch_of(json!({"a": 2, "b": [3]})), "", &mut log);
        assert_eq!(target, json!({"a": 2, "b": [3], "keep": "x"}));
        assert!(log.overrides("/a"));
        assert!(log.overrides("/b"));
        // A scalar set at an ancestor overrides descendants.
        assert!(log.overrides("/b/0"));
        assert!(!log.overrides("/keep"));
    }

    #[test]
    fn merge_objects_recursively() {
        let mut target = json!({"gen": {"temp": 1.0, "top_p": 0.9}});
        let mut log = MergeLog::new();
        merge_patch(
            &mut target,
            &patch_of(json!({"gen": {"temp": 0.5, "new": true}})),
            "",
            &mut log,
        );
        assert_eq!(target, json!({"gen": {"temp": 0.5, "top_p": 0.9, "new": true}}));
        assert!(log.overrides("/gen/temp"));
        // The object merge at /gen does NOT override its untouched children.
        assert!(!log.overrides("/gen/top_p"));
        // Nor does an object merge count as an ancestor override.
        assert!(!log.overrides("/gen/other"));
    }

    #[test]
    fn null_deletes_at_any_depth() {
        let mut target = json!({"a": {"b": 1, "c": 2}, "d": 3});
        let mut log = MergeLog::new();
        merge_patch(&mut target, &patch_of(json!({"a": {"b": null}, "d": null})), "", &mut log);
        assert_eq!(target, json!({"a": {"c": 2}}));
        assert!(log.overrides("/a/b"));
        assert!(log.overrides("/d"));
        // Deletion at an ancestor overrides descendants.
        assert!(log.overrides("/d/x"));
    }

    #[test]
    fn object_over_non_object_rebuilds() {
        let mut target = json!({"a": 5});
        let mut log = MergeLog::new();
        merge_patch(&mut target, &patch_of(json!({"a": {"x": 1}})), "", &mut log);
        assert_eq!(target, json!({"a": {"x": 1}}));
        // Exact pointer overridden (subtree rebuilt)…
        assert!(log.overrides("/a"));
        // …but per § 6 an object at an ancestor is not an ancestor override.
        assert!(!log.overrides("/a/y"));
        // The set leaf itself is.
        assert!(log.overrides("/a/x"));
    }

    #[test]
    fn pointer_escaping() {
        let mut target = json!({});
        let mut log = MergeLog::new();
        merge_patch(&mut target, &patch_of(json!({"a/b": 1, "c~d": 2})), "", &mut log);
        assert!(log.overrides("/a~1b"));
        assert!(log.overrides("/c~0d"));
    }

    #[test]
    fn from_unknown_drops_nulls() {
        let mut fields = Map::new();
        fields.insert("keep".into(), json!(1));
        fields.insert("drop".into(), Value::Null);
        let extra = Extra::from_unknown("fmt", fields);
        assert_eq!(extra.get("fmt").unwrap().len(), 1);
        assert!(extra.get("fmt").unwrap().contains_key("keep"));

        let empty = Extra::from_unknown("fmt", Map::new());
        assert!(empty.is_empty());
    }

    #[test]
    fn merge_into_only_touches_own_namespace() {
        let mut extra = Extra::new();
        extra.set("a_fmt", "x", 1);
        extra.set("b_fmt", "y", 2);
        let mut target = json!({});
        let mut log = MergeLog::new();
        extra.merge_into("a_fmt", &mut target, "", &mut log);
        assert_eq!(target, json!({"x": 1}));
    }

    #[test]
    fn extra_serde_round_trip() {
        let mut extra = Extra::new();
        extra.set("google_generate_content", "cachedContent", "cachedContents/abc");
        let s = serde_json::to_string(&extra).unwrap();
        assert_eq!(s, r#"{"google_generate_content":{"cachedContent":"cachedContents/abc"}}"#);
        let back: Extra = serde_json::from_str(&s).unwrap();
        assert_eq!(back, extra);
    }
}
